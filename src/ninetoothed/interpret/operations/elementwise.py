"""Element-wise operations: ``arith.*``, ``cmp.*``, ``math.*``, ``select.*``."""

from math import erf as _scalar_erf

import numpy as np

from ..dtypes import cast_array, is_integer_dtype, resolve_dtype
from ..errors import UnsupportedOperationError
from ..registry import register
from ..values import materialize
from .common import bind_data, numbers, operand, result_dtype

_BINARY = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.true_divide,
    "floordiv": np.floor_divide,
    "pow": np.power,
    # The `mod` operator is registered separately so it can use C-style semantics.
    "bitwise_left_shift": np.left_shift,
    "bitwise_right_shift": np.right_shift,
    "bitwise_and": np.bitwise_and,
    "bitwise_or": np.bitwise_or,
    "bitwise_xor": np.bitwise_xor,
    "and": np.bitwise_and,
    "or": np.bitwise_or,
    "maximum": np.maximum,
    "minimum": np.minimum,
}

_COMPARISONS = {
    "eq": np.equal,
    "ne": np.not_equal,
    "lt": np.less,
    "le": np.less_equal,
    "gt": np.greater,
    "ge": np.greater_equal,
}

_UNARY = {
    "neg": np.negative,
    "pos": np.positive,
    "not": np.logical_not,
    "invert": np.invert,
}

_MATH = {
    "abs": np.abs,
    "acos": np.arccos,
    "asin": np.arcsin,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "ceil": np.ceil,
    "cos": np.cos,
    "cosh": np.cosh,
    "erf": np.vectorize(_scalar_erf),
    "exp": np.exp,
    "exp2": np.exp2,
    "expm1": np.expm1,
    "floor": np.floor,
    "log": np.log,
    "log10": np.log10,
    "log1p": np.log1p,
    "log2": np.log2,
    "pow": np.power,
    "rsqrt": lambda value: 1.0 / np.sqrt(value),
    "sin": np.sin,
    "sinh": np.sinh,
    "sqrt": np.sqrt,
    "tan": np.tan,
    "tanh": np.tanh,
}


def _c_style_remainder(lhs, rhs):
    """Return the C-style remainder so integer results match the CUDA backend."""
    lhs_array = np.asarray(lhs)
    rhs_array = np.asarray(rhs)

    if is_integer_dtype(lhs_array.dtype) and is_integer_dtype(rhs_array.dtype):
        quotient = np.trunc(np.true_divide(lhs_array, rhs_array)).astype(
            lhs_array.dtype
        )

        return (lhs_array - quotient * rhs_array).astype(lhs_array.dtype, copy=False)

    return np.fmod(lhs, rhs)


def _apply_result_dtype(operation, result):
    dtype = result_dtype(operation, fallback=None)

    if dtype in {None, "bool", "none", "symbol", "dtype"}:
        return result

    return cast_array(result, resolve_dtype(dtype))


def _binary(state, operation, func):
    if _is_pointer(operation, state):
        _pointer_binary(state, operation)

        return

    lhs = materialize(operand(state, operation, 0), state.context)
    rhs = materialize(operand(state, operation, 1), state.context)

    with np.errstate(all="ignore"):
        result = func(lhs, rhs)

    bind_data(state, operation, _apply_result_dtype(operation, result))


def _is_pointer(operation, state):
    if operation.opcode.split(".", 1)[-1] not in {"add", "sub"}:
        return False

    return any(state.value(name).kind == "pointer" for name in operation.operands)


def _pointer_binary(state, operation):
    """Handle ``pointer + offset``, ``pointer - offset`` and ``pointer - pointer``."""
    from ..values import Pointer

    name = operation.opcode.split(".", 1)[-1]
    left = operand(state, operation, 0)
    right = operand(state, operation, 1)

    if left.kind == "pointer" and right.kind == "pointer":
        if name != "sub":
            raise UnsupportedOperationError(
                "Pointer addition is not defined between two pointers.",
                opcode=operation.opcode,
                location=state.location,
            )

        bind_data(state, operation, int(left.data.offset) - int(right.data.offset))

        return

    pointer = left if left.kind == "pointer" else right
    other = right if left.kind == "pointer" else left
    delta = materialize(other, state.context)

    if left.kind != "pointer" and name == "sub":
        raise UnsupportedOperationError(
            "Cannot subtract a pointer from a scalar.",
            opcode=operation.opcode,
            location=state.location,
        )

    if name == "sub":
        delta = -np.asarray(delta)

    if np.asarray(delta).size != 1:
        raise UnsupportedOperationError(
            "Pointer arithmetic requires a scalar offset.",
            opcode=operation.opcode,
            location=state.location,
        )

    bind_data(
        state,
        operation,
        Pointer(pointer.data.tensor, int(pointer.data.offset) + int(np.asarray(delta))),
    )


@register(
    *(f"arith.{name}" for name in _BINARY),
    category="arith",
    summary="Element-wise arithmetic, bitwise, and min/max operators.",
)
def _handle_binary(state, operation):
    name = operation.opcode.split(".", 1)[-1]

    if name in {"and", "or"}:
        left = operand(state, operation, 0)
        right = operand(state, operation, 1)

        if left.dtype == "bool" or right.dtype == "bool":
            _binary(
                state,
                operation,
                np.logical_and if name == "and" else np.logical_or,
            )

            return

    _binary(state, operation, _BINARY[name])


@register("arith.mod", category="arith", summary="C-style remainder (`%`).")
def _handle_mod(state, operation):
    _binary(state, operation, _c_style_remainder)


@register(
    *(f"arith.{name}" for name in _UNARY),
    category="arith",
    summary="Element-wise unary operators.",
)
def _handle_unary(state, operation):
    name = operation.opcode.split(".", 1)[-1]
    value = materialize(operand(state, operation, 0), state.context)

    with np.errstate(all="ignore"):
        result = _UNARY[name](value)

    bind_data(state, operation, _apply_result_dtype(operation, result))


@register(
    "arith.constant",
    category="arith",
    summary="Literal scalar constant (`int`, `float`, `bool`, `inf`).",
)
def _handle_constant(state, operation):
    value = operation.attrs.get("value")

    if isinstance(value, str):
        literals = {"inf": np.inf, "-inf": -np.inf, "nan": np.nan}

        if value not in literals:
            raise UnsupportedOperationError(
                f"Unsupported constant literal `{value}`.",
                opcode=operation.opcode,
                location=state.location,
            )

        value = literals[value]

    if value is None:
        bind_data(state, operation, 0)

        return

    bind_data(state, operation, np.asarray(value))


@register(
    *(f"cmp.{name}" for name in _COMPARISONS),
    category="cmp",
    summary="Element-wise comparison returning `bool`.",
)
def _handle_comparison(state, operation):
    name = operation.opcode.split(".", 1)[-1]
    lhs, rhs = numbers(state, operation)
    bind_data(state, operation, _COMPARISONS[name](lhs, rhs))


@register(
    *(f"math.{name}" for name in _MATH),
    category="math",
    summary="Element-wise math function.",
)
def _handle_math(state, operation):
    name = operation.opcode.split(".", 1)[-1]
    func = _MATH.get(name)

    if func is None:
        raise UnsupportedOperationError(
            f"Unsupported math function `{name}`.",
            opcode=operation.opcode,
            location=state.location,
        )

    args = numbers(state, operation)

    if not args:
        raise UnsupportedOperationError(
            f"Math function `{name}` requires at least one operand.",
            opcode=operation.opcode,
            location=state.location,
        )

    try:
        with np.errstate(all="ignore"):
            result = func(*args)
    except Exception as exc:  # noqa: BLE001 - surfaced as an interpreter error
        raise UnsupportedOperationError(
            f"Math function `{name}` failed: {exc}.",
            opcode=operation.opcode,
            location=state.location,
        ) from exc

    bind_data(state, operation, _apply_result_dtype(operation, result))


@register(
    "select.where",
    category="select",
    summary="Element-wise selection between two values.",
)
def _handle_where(state, operation):
    condition = materialize(operand(state, operation, 0), state.context)
    yes = materialize(operand(state, operation, 1), state.context)
    no = materialize(operand(state, operation, 2), state.context)
    result = np.where(np.asarray(condition, dtype=bool), yes, no)

    bind_data(state, operation, _apply_result_dtype(operation, result))


__all__ = []
