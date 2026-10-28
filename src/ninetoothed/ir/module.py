"""Immutable, target-neutral contract at the frontend/compiler boundary."""

import keyword
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace

from ninetoothed.dtype import normalize_dtype
from ninetoothed.ir import ssa
from ninetoothed.ir.frozen import freeze
from ninetoothed.ir.kernel import AccessMap, IndexExpr, TensorSpec
from ninetoothed.ir.operations import type_condition_value


@dataclass(frozen=True, kw_only=True)
class FrontendModule:
    """Freeze computation, layouts and specialization domains without Python objects."""

    name: str
    tensors: tuple[TensorSpec, ...]
    program: ssa.Program
    symbols: tuple[str, ...]
    symbol_roles: Mapping = field(default_factory=dict)
    bindings: Mapping = field(default_factory=dict)
    parameter_defaults: Mapping = field(default_factory=dict)
    named_layout: Mapping | None = None
    meta_domains: Mapping = field(default_factory=dict)
    constexpr_values: Mapping = field(default_factory=dict)
    source: str = ""
    version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "tensors", tuple(self.tensors))
        object.__setattr__(self, "symbols", tuple(self.symbols))

        for name in (
            "symbol_roles",
            "bindings",
            "parameter_defaults",
            "named_layout",
            "meta_domains",
            "constexpr_values",
        ):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    @property
    def public_args(self):
        return tuple(spec.name for spec in self.tensors)

    @property
    def meta_defaults(self):
        defaults = dict(self.parameter_defaults) | {
            name: domain["default"] for name, domain in self.meta_domains.items()
        }

        return {
            name: self.bindings.get(name, value) for name, value in defaults.items()
        }


def verify_module(module):
    """Check the neutral module before specialization and backend passes."""
    if not isinstance(module, FrontendModule) or module.version != 1:
        raise ssa.VerificationError("Unsupported frontend module version.")

    _verify_data(module)
    validate_identifier(module.name)

    for name in (*module.public_args, *module.symbols):
        validate_identifier(name)

    if set(module.program.metadata) - {
        "source",
        "function",
        "coarse_operator_nodes",
        "ssa_operation_count",
        "symbols",
        "shape_validation",
        "external_calls",
    }:
        raise ssa.VerificationError(
            "Frontend modules cannot carry backend-owned metadata."
        )

    ssa.verify_program(module.program, strict_types=True, frontend_layout=True)

    if module.program.blocks[0].args:
        raise ssa.VerificationError(
            "Frontend module entry arguments must be declared as public inputs."
        )

    names = module.public_args

    if (
        len(set(names)) != len(names)
        or tuple(value.name for value in module.program.inputs) != names
    ):
        raise ssa.VerificationError("Module tensor and SSA parameter bindings differ.")

    declared = set(module.symbols)

    if declared & set(names) or any(
        name in {"index", "outer_index"}
        or re.fullmatch(r"value_\d+|extract_\d+_\d+", name)
        for name in declared
    ):
        raise ssa.VerificationError(
            "Module symbols collide with parameter or coordinate identities."
        )

    if (
        len(declared) != len(module.symbols)
        or not set(module.program.metadata.get("symbols", ())) <= declared
    ):
        raise ssa.VerificationError("Module has an invalid explicit symbol table.")

    if set(module.symbol_roles) != declared or any(
        role not in {"index", "extent", "stride", "meta"}
        for role in module.symbol_roles.values()
    ):
        raise ssa.VerificationError("Module symbols require explicit index roles.")

    def check_block(block, visible, type_conditions=()):
        visible = dict(visible) | {value.name: value.type for value in block.args}
        type_conditions = dict(type_conditions)

        for value in block.args:
            _verify_value_type(value.type)

        for op in block.operations:
            if op.opcode == "dtype.select" and op.operands[0] not in type_conditions:
                raise ssa.VerificationError(
                    "Dtype selection must depend only on type information."
                )

            if op.opcode == "dtype.select":
                flag = type_conditions[op.operands[0]]
                yes, no = (visible[name] for name in op.operands[1:])
                expected = (
                    (yes if flag else no).dtype
                    if flag is not None
                    else yes.dtype
                    if yes.dtype == no.dtype
                    else None
                )

                if op.results[0].type.dtype != expected:
                    raise ssa.VerificationError(
                        "Dtype selection result differs from its type condition."
                    )

            if (
                op.opcode == "arith.constant"
                and type(op.attrs.get("value")) is bool
                or op.opcode in {"cmp.eq", "cmp.ne"}
                and all(visible[name].kind == "dtype" for name in op.operands)
                or op.opcode in {"arith.and", "arith.or", "arith.not"}
                and all(name in type_conditions for name in op.operands)
            ):
                operands = tuple(
                    ssa.Value(name=name, type=visible[name]) for name in op.operands
                )
                known = type_condition_value(op, operands, type_conditions)
                type_conditions.update((value.name, known) for value in op.results)

            for value in op.results:
                _verify_value_type(value.type)

                for text in value.type.shape:
                    if text in visible and text.startswith("%"):
                        type_ = visible[text]

                        if (
                            type_.shape
                            or type_.kind not in {"scalar", "index"}
                            or normalize_dtype(type_.dtype)
                            not in {None, "index", "int32", "int64"}
                        ):
                            raise ssa.VerificationError(
                                "SSA shape references must be integer scalars."
                            )
                    else:
                        _verify_expression(
                            IndexExpr.parse(text), declared, roles=module.symbol_roles
                        )

            for region in op.regions:
                check_block(region, visible, type_conditions)

            for name in ("dtype_ref", "dtype_operand", "preserve_dtype"):
                if op.attrs.get(name) is not None and op.attrs[name] not in visible:
                    raise ssa.VerificationError(
                        f"Undefined module type binding at {op.opcode}."
                    )

            required = {
                "arith.constant": "value",
                "shape.dim": "dim",
                "tensor.stride": "dim",
                "tensor.view": "subscript",
                "symbol.attr": "expr",
            }.get(op.opcode)

            if required is not None and required not in op.attrs:
                raise ssa.VerificationError(
                    f"Missing {required!r} attribute at {op.opcode}."
                )

            if op.opcode == "symbol.attr":
                _verify_expression(IndexExpr.parse(op.attrs["expr"]), declared)

            visible.update((value.name, value.type) for value in op.results)

    for block in module.program.blocks:
        check_block(block, {value.name: value.type for value in module.program.inputs})

    for spec, value in zip(module.tensors, module.program.inputs):
        _verify_value_type(value.type)
        shape = tuple(spec.attrs.get("application_shape", spec.shape))

        source_rank = spec.attrs.get("source_ndim", spec.ndim)

        if (
            type(spec.ndim) is not int
            or spec.ndim < 0
            or len(spec.shape) != spec.ndim
            or type(source_rank) is not int
            or source_rank < 0
            or len(spec.attrs.get("source_shape", ())) != source_rank
            or len(spec.attrs.get("source_strides", ())) != source_rank
            or type(spec.constexpr) is not bool
            or (
                spec.jagged_dim is not None
                and (
                    type(spec.jagged_dim) is not int
                    or not 0 <= spec.jagged_dim < source_rank
                )
            )
            or normalize_dtype(spec.attrs.get("source_dtype"))
            != normalize_dtype(spec.dtype)
            or normalize_dtype(spec.dtype) not in _DTYPES | {None}
        ):
            raise ssa.VerificationError(
                f"Invalid module tensor schema for {spec.name!r}."
            )

        if (
            value.type.kind != ("tensor" if spec.ndim else "scalar")
            or value.type.shape != shape
            or normalize_dtype(value.type.dtype) != normalize_dtype(spec.dtype)
        ):
            raise ssa.VerificationError(
                f"Module parameter type differs for {spec.name!r}."
            )

        input_attrs = {
            "ndim": spec.ndim,
            "constexpr": spec.constexpr,
            "jagged_dim": spec.jagged_dim,
            "dtype_level": 0,
        } | dict(spec.attrs)

        if dict(value.type.attrs) != input_attrs:
            raise ssa.VerificationError(
                f"Module parameter layout provenance differs for {spec.name!r}."
            )

        jagged_fields = tuple(
            f"jagged_{name}_param"
            for name in (
                "values",
                "values_numel",
                "offsets",
                "offsets_numel",
                "max_seq_len",
                "seq_len",
            )
        )
        jagged_names = tuple(spec.attrs.get(name) for name in jagged_fields)

        if (
            spec.jagged_dim is None
            and any(name in spec.attrs for name in jagged_fields)
        ) or (
            spec.jagged_dim is not None
            and (
                any(
                    type(name) is not str or name not in declared
                    for name in jagged_names
                )
                or len(set(jagged_names)) != len(jagged_names)
            )
        ):
            raise ssa.VerificationError(
                f"Invalid jagged auxiliary bindings for {spec.name!r}."
            )

        if (
            spec.jagged_dim is None
            and tuple(spec.attrs.get("view_shape", spec.shape)) != spec.shape
        ):
            raise ssa.VerificationError(
                f"Module view rank or shape differs for {spec.name!r}."
            )

        expressions = (*spec.shape, *spec.attrs.get("source_shape", ()), *shape)

        for text in expressions:
            _verify_expression(
                IndexExpr.parse(text), declared, roles=module.symbol_roles
            )

        for attribute, allowed in (
            ("source_shape", {"extent", "meta"}),
            ("source_strides", {"stride", "extent", "meta"}),
        ):
            for text in spec.attrs.get(attribute, ()):

                def check_role(expression):
                    if (
                        expression.op == "symbol"
                        and module.symbol_roles[expression.value] not in allowed
                    ):
                        raise ssa.VerificationError(
                            f"Module symbol role differs from {attribute!r}."
                        )

                    for operand in expression.operands:
                        check_role(operand)

                check_role(IndexExpr.parse(text))
                _verify_expression(IndexExpr.parse(text), declared)

        if spec.layout is not None:
            _verify_layout(
                spec.layout,
                spec.attrs,
                declared,
                roles=module.symbol_roles,
                jagged_dim=spec.jagged_dim,
            )

    if set(module.constexpr_values) - set(names) or set(module.meta_domains) - declared:
        raise ssa.VerificationError(
            "Module contains undeclared constant or meta bindings."
        )

    for name, domain in module.meta_domains.items():
        low, high, default = (domain[key] for key in ("lower", "upper", "default"))

        if (
            module.symbol_roles[name] != "meta"
            or any(type(item) is not int for item in (low, high, default))
            or not low <= default <= high
            or type(domain["power_of_two"]) is not bool
            or domain.get("scope") not in {"hard_layout", "triton_tuning"}
            or (domain["power_of_two"] and (default <= 0 or default & (default - 1)))
        ):
            raise ssa.VerificationError(f"Invalid meta domain for {name!r}.")

    if {name for name, role in module.symbol_roles.items() if role == "meta"} != set(
        module.meta_domains
    ):
        raise ssa.VerificationError("Module meta symbols require bounded domains.")

    validate_bindings(replace(module, bindings={}), module.parameter_defaults)
    validate_bindings(module, module.bindings)

    from ninetoothed.ir.named_layout import verify_named_layout

    verify_named_layout(module)

    return module


def validate_bindings(module, values):
    """Check supplied index values against their explicit roles and domains."""
    if set(values) - set(module.symbols):
        raise ssa.VerificationError(
            "Specialization contains undeclared module bindings."
        )

    for name, value in values.items():
        if name in module.bindings and value != module.bindings[name]:
            raise ssa.VerificationError(
                f"Cannot rebind an already specialized symbol {name!r}."
            )

        role = module.symbol_roles[name]

        if type(value) is not int or (role == "extent" and value < 0):
            raise ssa.VerificationError(
                f"Specialization value for {role} {name!r} must be a valid integer."
            )

        domain = module.meta_domains.get(name)

        if domain is not None and (
            not domain["lower"] <= value <= domain["upper"]
            or (domain["power_of_two"] and (value <= 0 or value & (value - 1)))
        ):
            raise ssa.VerificationError(
                f"Specialization violates meta domain for {name!r}."
            )


def _verify_layout(layout, attrs, declared, *, roles, jagged_dim=None):
    pairs = (
        (layout.source_shape, "source_shape"),
        (layout.source_strides, "source_strides"),
        (layout.view_shape, "view_shape"),
        (layout.application_shape, "application_shape"),
    )

    for values, name in pairs:
        if values != tuple(IndexExpr.parse(text) for text in attrs.get(name, ())):
            raise ssa.VerificationError(f"Structured layout differs from {name!r}.")

        if name != "source_strides":
            for expression in values:
                _verify_expression(expression, declared, roles=roles)

    if len(layout.source_shape) != len(layout.source_strides):
        raise ssa.VerificationError("Source layout rank and strides differ.")

    if tuple(level.shape for level in layout.levels) != tuple(
        tuple(IndexExpr.parse(text) for text in shape)
        for shape in attrs.get("dtype_shapes", ())
    ):
        raise ssa.VerificationError("Structured layout dtype levels differ.")

    for level in layout.levels:
        for expression in level.shape:
            _verify_expression(expression, declared, roles=roles)

    if tuple(level.target_dims for level in layout.levels) != tuple(
        tuple(None if dim is None else IndexExpr.parse(dim) for dim in dims)
        for dims in attrs.get("dtype_target_dims", ())
    ):
        raise ssa.VerificationError("Structured layout target dimensions differ.")

    def access_map(offsets, linear, mask):
        return AccessMap(
            source_indices=tuple(IndexExpr.parse(text) for text in offsets),
            linear_index=IndexExpr.parse(linear),
            predicate=IndexExpr.parse(mask),
        )

    view_access = (
        None
        if attrs.get("view_linear_offset") is None
        else access_map(
            attrs.get("view_offsets", ()),
            attrs["view_linear_offset"],
            attrs.get("view_mask", True),
        )
    )
    value_accesses = tuple(
        access_map(
            item.get("offsets", ()),
            item.get("linear_offset", 0),
            item.get("mask", True),
        )
        for item in attrs.get("access_templates", ())
    )

    if layout.view_access != view_access or layout.value_accesses != value_accesses:
        raise ssa.VerificationError(
            "Structured layout and legacy access descriptors differ."
        )

    for template in attrs.get("access_templates", ()):
        level = template.get("level")

        if (
            type(level) is not int
            or not 0 <= level < len(layout.levels)
            or tuple(IndexExpr.parse(text) for text in template.get("shape", ()))
            != layout.levels[level].shape
        ):
            raise ssa.VerificationError(
                "Access template level and shape differ from the layout."
            )

    coordinates = {"index", "outer_index"}
    coordinates.update(
        f"value_{dim}"
        for dim in range(len(layout.levels[-1].shape) if layout.levels else 0)
    )
    coordinates.update(
        f"extract_{level}_{dim}"
        for level, shape in enumerate(layout.levels[:-1])
        for dim in range(len(shape.shape))
    )

    for template in attrs.get("access_templates", ()):
        jagged = template.get("jagged")

        if jagged_dim is None:
            if "jagged" in template:
                raise ssa.VerificationError(
                    "Non-jagged access templates cannot declare jagged addressing."
                )

            continue

        if not isinstance(jagged, Mapping) or set(jagged) != {
            "offsets_param",
            "seq_start",
            "seq_len",
            "batch_offset",
            "stride",
        }:
            raise ssa.VerificationError("Incomplete jagged access template.")

        validate_identifier(jagged["seq_start"])

        if (
            jagged["offsets_param"] != attrs["jagged_offsets_param"]
            or jagged["seq_len"] != attrs["jagged_seq_len_param"]
            or jagged["seq_start"] in declared | coordinates
            or IndexExpr.parse(jagged["batch_offset"])
            != IndexExpr.parse(template["offsets"][0])
            or IndexExpr.parse(jagged["stride"])
            != IndexExpr.parse(attrs["source_strides"][jagged_dim])
        ):
            raise ssa.VerificationError(
                "Jagged access provenance differs from the physical layout."
            )

        for key in ("batch_offset", "stride"):
            _verify_expression(
                IndexExpr.parse(jagged[key]), declared, coordinates=coordinates
            )

    for expression in layout.source_strides:
        _verify_expression(expression, declared)

    for level in layout.levels:
        for expression in level.target_dims:
            if expression is not None:
                _verify_expression(expression, declared)

    for access in (layout.view_access, *layout.value_accesses):
        if access is not None and len(access.source_indices) != len(
            layout.source_shape
        ):
            raise ssa.VerificationError(
                "Access map source rank differs from the layout."
            )

        if access is not None:
            for expression in (*access.source_indices, access.linear_index):
                _verify_expression(expression, declared, coordinates=coordinates)

            _verify_expression(
                access.predicate, declared, coordinates=coordinates, predicate=True
            )


def _verify_expression(
    expression, declared, *, coordinates=(), roles=None, predicate=False, quotient=False
):
    try:
        IndexExpr.parse(expression.render())
    except (SyntaxError, ValueError, IndexError) as exc:
        raise ssa.VerificationError("Malformed module index expression.") from exc

    if expression.op == "symbol" and expression.value not in declared:
        if expression.value not in coordinates:
            raise ssa.VerificationError(
                f"Undeclared module symbol {expression.value!r}."
            )

    if (
        expression.op == "symbol"
        and roles is not None
        and roles.get(expression.value) not in {"extent", "meta"}
    ):
        raise ssa.VerificationError(
            "Module symbol role is not valid in a shape expression."
        )

    arities = {
        "constant": 0,
        "symbol": 0,
        "neg": 1,
        "pos": 1,
        "invert": 1,
        "not": 1,
        "add": 2,
        "sub": 2,
        "mul": 2,
        "div": 2,
        "floordiv": 2,
        "mod": 2,
        "pow": 2,
        "and": 2,
        "or": 2,
        "bitand": 2,
        "bitor": 2,
        "bitxor": 2,
        "eq": 2,
        "ne": 2,
        "lt": 2,
        "le": 2,
        "gt": 2,
        "ge": 2,
    }
    arity = (
        {"floor": 1, "Mod": 2}.get(expression.value)
        if expression.op == "call"
        else arities.get(expression.op)
    )

    if arity is None or len(expression.operands) != arity:
        raise ssa.VerificationError(
            "Unsupported module index expression operation or callee."
        )

    if (
        expression.op == "call"
        and expression.value == "floor"
        and expression.operands[0].op != "div"
    ):
        raise ssa.VerificationError("Layout floor requires an explicit quotient.")

    if expression.op == "pow" and (
        expression.operands[1].op != "constant"
        or type(expression.operands[1].value) is not int
        or not 0 <= expression.operands[1].value <= 8
    ):
        raise ssa.VerificationError(
            "Integer layout powers require a constant exponent from zero to eight."
        )

    if not predicate and expression.op in {
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "and",
        "or",
        "not",
    }:
        raise ssa.VerificationError("Boolean operations require a predicate context.")

    if expression.op == "div" and not quotient:
        raise ssa.VerificationError(
            "Integer layouts require floor division or an explicit floor quotient."
        )

    if expression.op == "constant" and (
        type(expression.value) not in {int, bool, float}
        or (not predicate and type(expression.value) is not int)
    ):
        raise ssa.VerificationError("Invalid module index constant.")

    for operand in expression.operands:
        _verify_expression(
            operand,
            declared,
            coordinates=coordinates,
            roles=roles,
            predicate=predicate,
            quotient=expression.op == "call" and expression.value == "floor",
        )


_DTYPES = {
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "float8_e4m3fn",
    "float8_e5m2",
}


def _verify_value_type(type_):
    if (
        type_.kind
        not in {
            "scalar",
            "tensor",
            "pointer",
            "index",
            "symbol",
            "none",
            "tuple",
            "dtype",
        }
        or normalize_dtype(type_.dtype)
        not in _DTYPES | {None, "index", "none", "symbol"}
        or (
            normalize_dtype(type_.dtype) in {"none", "symbol"}
            and type_.kind != "scalar"
        )
        or (
            type_.kind in {"scalar", "index", "symbol", "none", "tuple", "dtype"}
            and type_.shape
        )
    ):
        raise ssa.VerificationError("Invalid module SSA value type.")


def _verify_data(value):
    if value is None or type(value) in (str, int, float, bool):
        return

    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ssa.VerificationError("Frontend module mapping keys must be strings.")

        values = value.values()
    elif isinstance(value, tuple):
        values = value
    elif is_dataclass(value):
        if not value.__dataclass_params__.frozen:
            raise ssa.VerificationError(
                "Frontend modules require immutable IR records."
            )

        values = tuple(getattr(value, item.name) for item in fields(value))
    else:
        raise ssa.VerificationError(
            "Frontend modules must contain only immutable IR data."
        )

    for item in values:
        _verify_data(item)


__all__ = ["FrontendModule", "verify_module", "validate_bindings"]


def validate_identifier(name):
    if (
        not isinstance(name, str)
        or re.fullmatch(r"[A-Za-z_]\w*", name, flags=re.ASCII) is None
        or keyword.iskeyword(name)
    ):
        raise ssa.VerificationError(f"Invalid compiler identifier {name!r}.")


def evaluate_index_expression(expression, symbols):
    """Evaluate the closed integer layout grammar, without executing source text."""
    import operator
    from fractions import Fraction

    if expression.op == "symbol":
        return symbols[expression.value]

    if expression.op == "constant":
        return expression.value

    values = tuple(
        evaluate_index_expression(item, symbols) for item in expression.operands
    )
    unary = {
        "neg": operator.neg,
        "pos": operator.pos,
        "invert": operator.invert,
        "not": operator.not_,
    }
    binary = {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "div": lambda lhs, rhs: Fraction(lhs, rhs),
        "floordiv": operator.floordiv,
        "mod": operator.mod,
        "pow": operator.pow,
        "bitand": operator.and_,
        "bitor": operator.or_,
        "bitxor": operator.xor,
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
        "and": lambda lhs, rhs: lhs and rhs,
        "or": lambda lhs, rhs: lhs or rhs,
    }

    if expression.op in unary:
        return unary[expression.op](*values)

    if expression.op in binary:
        result = binary[expression.op](*values)
    elif expression.op == "call" and expression.value == "floor":
        result = values[0].__floor__()
    elif expression.op == "call" and expression.value == "Mod":
        result = values[0] % values[1]
    else:
        raise ValueError("Unsupported physical layout expression.")
    return result
