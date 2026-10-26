"""Backend-neutral symbolic specialization for runtime-known Launch IR values."""

import ast
import re
from dataclasses import fields, is_dataclass, replace
from typing import Any, Mapping

from ninetoothed.dtype import normalize_dtype
from ninetoothed.errors import LoweringError
from ninetoothed.ir import IndexExpr, Kernel, TensorSpec, ssa
from ninetoothed.ir.instantiation import instantiate_program
from ninetoothed.ir.module import validate_bindings, verify_module
from ninetoothed.naming import remove_prefixes


def specialize_tensor_specs(
    specs: tuple[TensorSpec, ...], values: Mapping[str, Any]
) -> tuple[TensorSpec, ...]:
    result = tuple(
        replace(
            spec,
            shape=tuple(_specialize_value(axis, values) for axis in spec.shape),
            attrs=_specialize_tensor_attrs(spec, values),
            layout=_specialize_layout(spec.layout, values),
        )
        for spec in specs
    )

    return result


def _specialize_layout(value, values):
    """Substitute layout expressions without rewriting axis identities."""
    if isinstance(value, IndexExpr):
        result = _substitute_index_expr(value, values)

        return IndexExpr.parse(_evaluate_constant_expression(result.render()))

    if isinstance(value, tuple):
        return tuple(_specialize_layout(item, values) for item in value)

    if is_dataclass(value):
        return replace(
            value,
            **{
                field.name: _specialize_layout(getattr(value, field.name), values)
                for field in fields(value)
            },
        )

    return value


def _substitute_index_expr(expression, values):
    # Textual and structured layouts must fold at the same expression boundary.
    if expression.op == "symbol" and expression.value in values:
        return IndexExpr.parse(values[expression.value])
    return replace(
        expression,
        operands=tuple(
            _substitute_index_expr(item, values) for item in expression.operands
        ),
    )


def _specialize_tensor_attrs(spec, values):
    """Substitute expression nodes, never logical axis or parameter names."""
    return _specialize_contract_attrs(spec.attrs, values, spec.name)


def _specialize_contract_attrs(attributes, values, parameter):
    attrs = {
        key: item
        if (key.startswith("jagged_") and key.endswith("_param"))
        or key in {"source_name", "layout_operations", "custom_tiling"}
        else _specialize_value(item, values)
        for key, item in attributes.items()
    }

    if "access_templates" in attributes:
        attrs["access_templates"] = tuple(
            dict(specialized)
            | {
                "jagged": dict(specialized["jagged"])
                | {
                    key: template["jagged"][key]
                    for key in ("offsets_param", "seq_start", "seq_len")
                }
            }
            if "jagged" in template
            else specialized
            for template, specialized in zip(
                attributes["access_templates"], attrs["access_templates"]
            )
        )

    return attrs


def specialize_program(program: ssa.Program, values: Mapping[str, Any]) -> ssa.Program:
    return replace(
        program,
        inputs=tuple(_specialize_ssa_value(value, values) for value in program.inputs),
        outputs=tuple(
            _specialize_ssa_value(value, values) for value in program.outputs
        ),
        blocks=tuple(_specialize_block(block, values) for block in program.blocks),
        metadata=program.metadata,
    )


def specialize_module(module, *, values=None, dtypes=None):
    """Bind a verified module without invoking an arrangement or Python parser."""
    verify_module(module)
    values, dtypes = dict(values or {}), dict(dtypes or {})

    if set(dtypes) - set(module.public_args):
        raise LoweringError("Specialization contains undeclared module bindings.")

    validate_bindings(module, values)

    tensors = (
        specialize_tensor_specs(module.tensors, values) if values else module.tensors
    )
    program = specialize_program(module.program, values) if values else module.program

    if dtypes:
        supported = {
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

        if any(normalize_dtype(dtype) not in supported for dtype in dtypes.values()):
            raise LoweringError("Specialization contains an unsupported element dtype.")

        tensors = tuple(
            replace(
                spec,
                dtype=normalize_dtype(dtypes.get(spec.name, spec.dtype)),
                attrs=dict(spec.attrs)
                | {"source_dtype": normalize_dtype(dtypes.get(spec.name, spec.dtype))},
            )
            for spec in tensors
        )
        program = instantiate_program(program, tensors)

    specialized = replace(
        module,
        tensors=tensors,
        program=program,
        bindings=dict(module.bindings) | values,
    )

    return verify_module(specialized)


def scheduled_meta_defaults(
    defaults: Mapping[str, int],
    schedule: Mapping[str, Any],
    *,
    domains=None,
    bindings=None,
) -> dict[str, int]:
    """Apply a selected schedule's tile values to matching meta parameters."""
    result = dict(defaults)
    tile = dict(schedule.get("tile", {}))
    bindings = dict(bindings or {})

    for name in result:
        if name in bindings:
            result[name] = bindings[name]
            continue

        schedule_name = _schedule_tile_name(name)

        if schedule_name in tile:
            value = tile[schedule_name]
            domain = (domains or {}).get(name)

            if (
                domain is not None
                and domain["scope"] == "hard_layout"
                and (
                    type(value) is not int
                    or not domain["lower"] <= value <= domain["upper"]
                    or (domain["power_of_two"] and (value <= 0 or value & (value - 1)))
                )
            ):
                raise LoweringError(
                    f"Backend schedule violates hard meta domain for {name!r}."
                )

            result[name] = int(value)
    return result


def specialize_schedule_tiles(kernel: Kernel) -> Kernel:
    """Materialize runtime symbols and backend tile choices before emission."""
    if kernel.ssa is None:
        return kernel

    schedule = dict(kernel.ssa.metadata.get("schedule", {}))
    defaults = dict(kernel.metadata.get("meta_defaults", {}))
    tile = dict(schedule.get("tile", {}))
    bindings = dict(kernel.metadata.get("bound_meta", {}))
    selected_defaults = scheduled_meta_defaults(
        defaults,
        schedule,
        domains=kernel.metadata.get("meta_domains", {}),
        bindings=bindings,
    )
    values = dict(kernel.metadata.get("specialization_values", {}))
    values.update(
        {
            name: selected_defaults[name]
            for name in defaults
            if (schedule_name := _schedule_tile_name(name)) in tile
        }
    )

    if not values:
        return kernel

    # A fixed logical tile cannot be overwritten by an optimization candidate.
    for name, value in bindings.items():
        schedule_name = _schedule_tile_name(name)

        if schedule_name in tile:
            tile[schedule_name] = value

    program = specialize_program(kernel.ssa, values)
    program = replace(
        program,
        metadata=dict(program.metadata) | {"schedule": schedule | {"tile": tile}},
    )

    return replace(
        kernel,
        tensors=specialize_tensor_specs(kernel.tensors, values),
        ssa=ssa.verify_program(program),
        metadata=dict(kernel.metadata) | {"meta_defaults": selected_defaults},
    )


def is_schedule_tile_parameter(name: str) -> bool:
    """Return whether a runtime meta value is owned by backend scheduling."""
    return _schedule_tile_name(name) in {"block_m", "block_n", "block_k"}


def _schedule_tile_name(name: str) -> str:
    return remove_prefixes(name).lower().replace("block_size_", "block_")


def _specialize_block(block: ssa.Block, values: Mapping[str, Any]) -> ssa.Block:
    return replace(
        block,
        args=tuple(_specialize_ssa_value(value, values) for value in block.args),
        operations=tuple(
            _specialize_operation(operation, values) for operation in block.operations
        ),
    )


def _specialize_operation(
    operation: ssa.Operation, values: Mapping[str, Any]
) -> ssa.Operation:
    return replace(
        operation,
        # SSA operands are identities, not textual index expressions. Symbols
        # retain their ABI bindings even when layout expressions are specialized.
        operands=operation.operands,
        results=tuple(
            _specialize_ssa_value(value, values) for value in operation.results
        ),
        attrs={
            key: item
            if key
            in {
                "axis",
                "target",
                "dtype",
                "dtype_ref",
                "subscript",
                "callee",
                "python",
                "source",
                "intrinsic",
                "induction",
                "iter_args",
                "indices",
                "bounds",
                "inclusive",
                "preserve_dtype",
            }
            else _specialize_value(item, values)
            for key, item in operation.attrs.items()
        },
        regions=tuple(
            _specialize_block(region, values) for region in operation.regions
        ),
    )


def _specialize_ssa_value(value: ssa.Value, values: Mapping[str, Any]) -> ssa.Value:
    return replace(value, type=_specialize_type(value.type, values))


def _specialize_type(type_: ssa.Type, values: Mapping[str, Any]) -> ssa.Type:
    return replace(
        type_,
        shape=tuple(_specialize_value(axis, values) for axis in type_.shape),
        attrs=_specialize_contract_attrs(type_.attrs, values, "SSA value"),
    )


def _specialize_value(value: Any, values: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return _specialize_text(value, values)

    if isinstance(value, tuple):
        return tuple(_specialize_value(item, values) for item in value)

    if isinstance(value, list):
        return [_specialize_value(item, values) for item in value]

    if isinstance(value, Mapping):
        return {key: _specialize_value(item, values) for key, item in value.items()}
    return value


def _specialize_text(text: str, values: Mapping[str, Any]) -> str:
    result = text

    for symbol in sorted(values, key=len, reverse=True):
        result = re.sub(
            rf"(?<!\w){re.escape(symbol)}(?!\w)",
            repr(values[symbol]),
            result,
        )

    if result == text:
        return text

    return _evaluate_constant_expression(result)


def _evaluate_constant_expression(text: str) -> str:
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return text

    if any(
        isinstance(node, (ast.Name, ast.Call, ast.Attribute)) for node in ast.walk(tree)
    ):
        return text

    try:
        value = eval(
            compile(tree, "<ssa-specialization>", "eval"), {"__builtins__": {}}
        )
    except (ArithmeticError, TypeError, ValueError):
        return text
    return repr(value) if isinstance(value, (bool, int, float)) else text


__all__ = [
    "is_schedule_tile_parameter",
    "scheduled_meta_defaults",
    "specialize_program",
    "specialize_module",
    "specialize_schedule_tiles",
    "specialize_tensor_specs",
]
