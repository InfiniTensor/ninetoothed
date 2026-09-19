"""Reduction and linear-algebra operations."""

import numpy as np

from ..dtypes import resolve_dtype
from ..registry import register
from ..values import materialize
from .common import bind_data, numbers, operand, result_dtype

_REDUCTIONS = {
    "reduce.sum": np.sum,
    "reduce.max": np.max,
    "reduce.min": np.min,
}

_ACCUMULATE_DTYPES = {
    "float8_e4m3fn",
    "float8_e5m2",
    "float16",
    "bfloat16",
}


@register(
    *_REDUCTIONS,
    category="reduce",
    summary="Reduction over one axis (or over all elements).",
)
def _handle_reduce(state, operation):
    value = materialize(operand(state, operation, 0), state.context)
    axis = operation.attrs.get("axis")
    dtype = result_dtype(operation, fallback=None)
    target = (
        None if dtype in {None, "none", "symbol", "dtype"} else resolve_dtype(dtype)
    )
    work = value

    if target is not None and target.kind == "f" and value.dtype.kind == "f":
        # Accumulate in the result dtype so `float32` matches the reference.
        work = value.astype(target)

    func = _REDUCTIONS[operation.opcode]

    if work.size == 0:
        result = np.asarray(0, dtype=work.dtype)
    else:
        result = func(work, axis=axis) if axis is not None else func(work)

    if target is not None:
        result = np.asarray(result).astype(target, copy=False)

    bind_data(state, operation, result)


def _dot_dtype(lhs_dtype, rhs_dtype):
    """Return the accumulator dtype for a dot product."""
    if str(lhs_dtype) in _ACCUMULATE_DTYPES or str(rhs_dtype) in _ACCUMULATE_DTYPES:
        return np.float32

    return np.promote_types(lhs_dtype, rhs_dtype)


@register(
    "linalg.dot",
    "linalg.matmul",
    category="linalg",
    summary="Matrix product with float32 accumulation for low-precision inputs.",
)
def _handle_dot(state, operation):
    lhs, rhs = numbers(state, operation)[:2]
    accumulator = _dot_dtype(lhs.dtype, rhs.dtype)

    with np.errstate(all="ignore"):
        result = np.matmul(
            lhs.astype(accumulator, copy=False), rhs.astype(accumulator, copy=False)
        )

    bind_data(state, operation, result)


@register(
    "linalg.transpose",
    category="linalg",
    summary="Transpose the last two dimensions of a tensor.",
)
def _handle_transpose(state, operation):
    value = materialize(operand(state, operation, 0), state.context)

    if value.ndim < 2:
        bind_data(state, operation, value)

        return

    bind_data(state, operation, np.swapaxes(value, -1, -2))


__all__ = []
