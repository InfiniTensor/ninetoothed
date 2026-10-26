"""Shared operation signatures checked before optimization and emission."""

import re
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class OperationSignature:
    """Describe operand arity and the number of SSA results."""

    minimum: int
    maximum: int | None
    results: int = 1


_SIGNATURES = {}


def register_operation(opcode, *, operands, results=1):
    """Register an explicit signature; existing semantics cannot be replaced."""
    if opcode in _SIGNATURES:
        raise ValueError(f"Operation {opcode!r} is already registered.")

    minimum, maximum = operands if isinstance(operands, tuple) else (operands, operands)

    if minimum < 0 or (maximum is not None and maximum < minimum) or results < 0:
        raise ValueError("Operation signature arities must be nonnegative.")

    _SIGNATURES[opcode] = OperationSignature(
        minimum=minimum, maximum=maximum, results=results
    )


def verify_operation_signature(operation, location, *, external_calls=()):
    """Reject unknown opcodes and malformed pure-operation signatures."""
    from ninetoothed.ir.ssa import VerificationError

    # Region and memory signatures are checked with their effect contracts.
    if operation.opcode.startswith(("scf.", "mem.")):
        return

    signature = _SIGNATURES.get(operation.opcode)

    if signature is None:
        declaration = (
            operation.opcode,
            operation.attrs.get("callee"),
            len(operation.operands),
        )

        if (
            operation.opcode.startswith("call.")
            and re.fullmatch(r"call\.[A-Za-z_]\w*", operation.opcode)
            and declaration in external_calls
            and len(operation.results) == 1
        ):
            return

        raise VerificationError(f"Unknown SSA operation at {location}.")

    count = len(operation.operands)

    if (
        count < signature.minimum
        or (signature.maximum is not None and count > signature.maximum)
        or len(operation.results) != signature.results
    ):
        raise VerificationError(f"Invalid operation signature at {location}.")


def verify_operation_types(
    operation, visible, location, *, strict=True, frontend_layout=False
):
    """Validate primitive result types using the same inference as the frontend."""
    import ast

    from ninetoothed.ir import ssa
    from ninetoothed.ir.type_inference import (
        TYPE_LAYOUT_ATTRS,
        _binary_type,
        _bool_type,
        _cast_type,
        _load_type,
        _math_result_type,
        _matmul_type,
        _offset_type,
        _reduce_type,
        _select_type,
        _subscript_type,
        _transpose_type,
    )

    integer_dtypes = {
        None,
        "index",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }

    if operation.opcode == "mem.store":
        indices = operation.attrs.get("indices", ())

        for index in (indices,) if isinstance(indices, str) else indices:
            type_ = visible[index]

            if type_.dtype not in integer_dtypes or type_.kind not in {
                "index",
                "scalar",
                "tensor",
            }:
                raise ssa.VerificationError(
                    f"Store indices must be integers at {location}."
                )

        if visible[operation.operands[0]].kind not in {"scalar", "index", "tensor"}:
            raise ssa.VerificationError(f"Invalid store value at {location}.")

        if strict:
            _verify_store(operation, visible, location)

    if len(operation.results) != 1:
        return

    values = tuple(
        ssa.Value(name=name, type=visible[name]) for name in operation.operands
    )
    code = operation.opcode
    name = code.split(".", 1)[1]
    binary = {
        "add": ast.Add,
        "sub": ast.Sub,
        "mul": ast.Mult,
        "div": ast.Div,
        "floordiv": ast.FloorDiv,
        "mod": ast.Mod,
        "pow": ast.Pow,
        "bitwise_and": ast.BitAnd,
        "bitwise_or": ast.BitOr,
        "bitwise_xor": ast.BitXor,
        "bitwise_left_shift": ast.LShift,
        "bitwise_right_shift": ast.RShift,
    }
    expected = None

    if code == "dtype.constant":
        from ninetoothed.dtype import normalize_dtype

        value = operation.attrs.get("value")

        if not isinstance(value, str):
            raise ssa.VerificationError(f"Invalid dtype constant at {location}.")

        expected = ssa.Type(kind="dtype", dtype=normalize_dtype(value))
    elif code == "dtype.of":
        if values[0].type.kind not in {"scalar", "index", "tensor", "pointer"}:
            raise ssa.VerificationError(f"Invalid dtype source at {location}.")

        expected = ssa.Type(kind="dtype", dtype=values[0].type.dtype)
    elif code == "dtype.select":
        condition, yes, no = (value.type for value in values)

        if (
            condition.kind != "scalar"
            or condition.dtype != "bool"
            or condition.shape
            or yes.kind != "dtype"
            or no.kind != "dtype"
        ):
            raise ssa.VerificationError(f"Invalid dtype selection at {location}.")

        expected = ssa.Type(
            kind="dtype", dtype=yes.dtype if yes.dtype == no.dtype else None
        )
    elif code.startswith("arith.") and name in binary:
        if any(value.type.kind == "dtype" for value in values):
            raise ssa.VerificationError(
                f"Dtypes cannot be used in arithmetic at {location}."
            )

        expected = _binary_type(binary[name](), *(value.type for value in values))
    elif code.startswith("cmp.") or code in {"arith.and", "arith.or", "arith.not"}:
        if any(value.type.kind == "dtype" for value in values) and (
            code not in {"cmp.eq", "cmp.ne"}
            or any(value.type.kind != "dtype" for value in values)
        ):
            raise ssa.VerificationError(f"Invalid dtype comparison at {location}.")

        expected = _bool_type(*values)
    elif code.startswith("math.") or code in {"arith.maximum", "arith.minimum"}:
        expected = _math_result_type(name, values)
    elif code == "select.where":
        expected = _select_type(*(value.type for value in values))
    elif code.startswith("reduce."):
        expected = _reduce_type(
            values[0].type, operation.attrs.get("axis"), operator=name, strict=strict
        )
    elif code in {"linalg.dot", "linalg.matmul"}:
        expected = _matmul_type(
            *(value.type for value in values), strict=strict and code == "linalg.matmul"
        )
    elif code == "linalg.transpose":
        expected = _transpose_type(values[0].type)
    elif code == "tensor.cast":
        ref = operation.attrs.get("dtype_ref")
        dtype = (
            visible[ref].dtype
            if ref in visible
            else values[1].type.dtype
            if len(values) == 2
            else operation.attrs.get("dtype")
        )
        expected = _cast_type(values[0].type, dtype)
    elif code in {"tensor.view", "tensor.extract"} and "subscript" in operation.attrs:
        subscript, count = parse_subscript_contract(operation.attrs["subscript"])

        if count != len(values) - 1:
            raise ssa.VerificationError(
                f"Subscript index bindings differ at {location}."
            )

        from ninetoothed.errors import LoweringError

        try:
            expected = _subscript_type(
                values[0].type,
                subscript,
                source=operation.attrs.get("source", False),
            )
        except LoweringError as exc:
            raise ssa.VerificationError(
                f"Invalid subscript at {location} ({exc})."
            ) from exc

        if any(
            value.type.kind not in {"scalar", "index", "tensor"}
            or value.type.dtype
            not in {
                None,
                "index",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
            }
            for value in values[1:]
        ):
            raise ssa.VerificationError(
                f"Extract indices must be integers at {location}."
            )
    elif code in {"arith.neg", "arith.pos", "arith.invert"}:
        expected = values[0].type
    elif code == "mem.load":
        expected = _load_type(values[0].type)
    elif code == "mem.data_ptr":
        expected = ssa.Type(kind="pointer", dtype=values[0].type.dtype)
    elif code == "mem.atomic_add":
        expected = ssa.Type(kind="scalar", dtype=values[1].type.dtype)
    elif code.startswith("call."):
        expected = values[0].type if values else ssa.Type(kind="tensor")
    elif code in {"shape.dim", "tensor.stride"}:
        dim = operation.attrs.get("dim")
        shape = (
            values[0].type.attrs.get("source_shape", values[0].type.shape)
            if operation.attrs.get("source") or code == "tensor.stride"
            else values[0].type.shape
        )

        if strict and (type(dim) is not int or not -len(shape) <= dim < len(shape)):
            raise ssa.VerificationError(f"Invalid dimension at {location}.")

        expected = ssa.Type(kind="index")
    elif code == "index.offset":
        expected = (
            ssa.Type(kind="index")
            if operation.attrs.get("decomposition") in {"matmul", "transpose"}
            else _offset_type(values[0].type, operation.attrs.get("dim"))
        )
    elif code == "tuple.construct" or code == "symbol.attr":
        expected = ssa.Type(kind="tuple" if code == "tuple.construct" else "symbol")
    elif code in {"tensor.zeros", "tensor.empty", "tensor.full"}:
        if strict and "shape_values" not in operation.attrs:
            raise ssa.VerificationError(
                f"Missing constructor shape contract at {location}."
            )

        if "shape_values" in operation.attrs:
            ref = operation.attrs.get("dtype_ref")
            dtype_operand = operation.attrs.get("dtype_operand")

            if dtype_operand is not None and (
                dtype_operand not in visible
                or visible[dtype_operand].kind != "dtype"
                or operation.operands[-1:] != (dtype_operand,)
            ):
                raise ssa.VerificationError(
                    f"Invalid constructor dtype operand at {location}."
                )

            expected = ssa.Type(
                kind="tensor",
                shape=operation.attrs["shape_values"],
                dtype=visible[ref].dtype
                if ref in visible
                else visible[dtype_operand].dtype
                if dtype_operand is not None
                else operation.attrs.get("element_dtype", "float32"),
            )
    elif code == "arith.constant":
        if "value" not in operation.attrs and not strict:
            return

        value = operation.attrs.get("value")
        actual = operation.results[0].type
        dtype = actual.dtype
        from ninetoothed.dtype import normalize_dtype

        dtype = normalize_dtype(dtype)
        valid = (
            actual.kind in {"scalar", "index"}
            and not actual.shape
            and (
                (type(value) is bool and dtype == "bool")
                or (type(value) is int and dtype in integer_dtypes - {None})
                or (
                    type(value) in {int, float}
                    and dtype in {"float16", "bfloat16", "float32", "float64"}
                )
                or (value in ("inf", "-inf") and dtype in {"float32", "float64"})
                or (value is None and dtype == "none")
                or (type(value) is str and dtype == "symbol")
            )
        )

        if not valid:
            raise ssa.VerificationError(f"Invalid constant type at {location}.")

    if expected is not None:
        if code.startswith("linalg.") and operation.attrs.get("dtype_ref"):
            from dataclasses import replace

            ref = operation.attrs["dtype_ref"]

            if ref not in visible:
                raise ssa.VerificationError(
                    f"Undefined linalg dtype binding at {location}."
                )

            expected = (
                replace(expected, dtype=visible[ref].dtype, attrs=visible[ref].attrs)
                if strict
                else visible[ref]
            )

        if operation.attrs.get("preserve_dtype"):
            from dataclasses import replace

            if operation.attrs["preserve_dtype"] != operation.operands[0]:
                raise ssa.VerificationError(
                    f"Invalid in-place dtype binding at {location}."
                )

            expected = replace(expected, dtype=values[0].type.dtype)

        ssa._verify_type(operation.results[0].type, expected, location)

        if strict and frontend_layout:
            if any(
                operation.results[0].type.attrs.get(key) != expected.attrs.get(key)
                for key in TYPE_LAYOUT_ATTRS
            ):
                raise ssa.VerificationError(
                    f"Semantic result layout differs at {location}."
                )


for _name in (
    "add",
    "sub",
    "mul",
    "div",
    "floordiv",
    "mod",
    "pow",
    "and",
    "or",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "maximum",
    "minimum",
):
    register_operation(f"arith.{_name}", operands=2)

for _name in ("neg", "pos", "not", "invert"):
    register_operation(f"arith.{_name}", operands=1)

for _name in ("eq", "ne", "lt", "le", "gt", "ge"):
    register_operation(f"cmp.{_name}", operands=2)

for _name in (
    "abs",
    "acos",
    "asin",
    "atan",
    "ceil",
    "cos",
    "cosh",
    "erf",
    "exp",
    "exp2",
    "expm1",
    "floor",
    "log",
    "log1p",
    "log2",
    "log10",
    "rsqrt",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
):
    register_operation(f"math.{_name}", operands=1)

for _name in ("atan2", "pow", "rand"):
    register_operation(f"math.{_name}", operands=2)

for _name in ("sum", "max", "min"):
    register_operation(f"reduce.{_name}", operands=1)

for _opcode, _arity in {
    "arith.constant": 0,
    "index.offset": 1,
    "shape.dim": 1,
    "tensor.stride": 1,
    "tensor.view": 1,
    "tensor.extract": (2, None),
    "tensor.cast": (1, 2),
    "tensor.zeros": (0, 1),
    "tensor.empty": (0, 1),
    "tensor.full": (1, 2),
    "dtype.constant": 0,
    "dtype.of": 1,
    "dtype.select": 3,
    "linalg.dot": 2,
    "linalg.matmul": 2,
    "linalg.transpose": 1,
    "select.where": 3,
    "tuple.construct": (0, None),
    "symbol.attr": 0,
    "call.fmod": 2,
}.items():
    register_operation(_opcode, operands=_arity)


def type_condition_value(operation, operands, conditions):
    """Evaluate a proven type-only boolean without executing frontend Python."""
    code = operation.opcode

    if code == "arith.constant" and type(operation.attrs.get("value")) is bool:
        return operation.attrs["value"]

    if code in {"cmp.eq", "cmp.ne"} and all(
        value.type.kind == "dtype" and value.type.dtype is not None
        for value in operands
    ):
        equal = operands[0].type.dtype == operands[1].type.dtype

        return equal if code == "cmp.eq" else not equal

    if code in {"arith.and", "arith.or", "arith.not"} and all(
        conditions.get(name) is not None for name in operation.operands
    ):
        values = [conditions[name] for name in operation.operands]

        if code == "arith.not":
            return not values[0]
        return all(values) if code == "arith.and" else any(values)

    return None


__all__ = [
    "OperationSignature",
    "register_operation",
    "verify_operation_signature",
    "verify_operation_types",
]


def parse_subscript_contract(text):
    """Parse the closed view descriptor; dynamic values live in SSA operands."""
    import ast

    from ninetoothed.ir.ssa import VerificationError

    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]

    try:
        node = ast.parse(f"value[{text}]", mode="eval").body.slice
    except (SyntaxError, ValueError) as exc:
        raise VerificationError("Malformed subscript contract.") from exc

    elements = node.elts if isinstance(node, ast.Tuple) else (node,)
    count = 0

    for position, item in enumerate(elements):
        if isinstance(item, ast.Slice):
            if any(part is not None for part in (item.lower, item.upper, item.step)):
                raise VerificationError(
                    "Only full slices are supported in local views."
                )
        elif isinstance(item, ast.Constant) and item.value in (None, Ellipsis):
            continue
        else:
            if not isinstance(item, ast.Name) or item.id != f"_index_{position}":
                raise VerificationError("Unsupported subscript index expression.")

            count += 1
    return node, count


def _verify_store(operation, visible, location):
    from ninetoothed.ir.ssa import VerificationError
    from ninetoothed.ir.type_inference import _broadcast_shapes, _subscript_type

    value, destination = (visible[name] for name in operation.operands)
    indices = operation.attrs.get("indices", ())
    indices = (indices,) if isinstance(indices, str) else tuple(indices)
    target = destination

    if "subscript" in operation.attrs:
        subscript, count = parse_subscript_contract(operation.attrs["subscript"])

        if count != len(indices):
            raise VerificationError(f"Store index bindings differ at {location}.")

        from ninetoothed.errors import LoweringError

        try:
            target = _subscript_type(
                destination,
                subscript,
                source=operation.attrs.get("source", False),
                index_types=tuple(visible[index] for index in indices),
            )
        except LoweringError as exc:
            raise VerificationError(
                f"Invalid store subscript at {location} ({exc})."
            ) from exc

        expected = {
            "target_shape": target.shape,
            "target_dtype_level": target.attrs.get("dtype_level", 0),
            "base_dtype_level": destination.attrs.get("dtype_level", 0),
        }

        if any(operation.attrs.get(name) != item for name, item in expected.items()):
            raise VerificationError(f"Store view contract differs at {location}.")
    elif indices:
        # Target-lowered scalar programs encode explicit physical coordinates.
        if operation.attrs.get("decomposition") not in {"matmul", "transpose"}:
            raise VerificationError(
                f"Store indices require a subscript contract at {location}."
            )
    elif any(
        name in operation.attrs
        for name in ("target_shape", "target_dtype_level", "base_dtype_level")
    ):
        raise VerificationError(
            f"Store view contract requires a subscript at {location}."
        )

    try:
        broadcast = _broadcast_shapes(target.shape, value.shape)
    except ValueError as exc:
        raise VerificationError(
            f"Store value shape cannot broadcast to target at {location}."
        ) from exc

    if broadcast != target.shape:
        raise VerificationError(
            f"Store value shape cannot broadcast to target at {location}."
        )
