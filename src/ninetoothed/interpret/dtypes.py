"""Dtype handling for the CPU reference interpreter.

The interpreter is a reference implementation, so dtypes must round-trip exactly
whenever the host can represent them.  Integer and boolean semantics are
bit-exact.  Floating point results are produced by NumPy in the requested width,
with no silent widening to ``float64``, so ``float32`` comparisons against NumPy
or PyTorch references stay within the documented tolerance.
"""

import numpy as np

from ninetoothed.dtype import normalize_dtype

from .errors import UnsupportedDTypeError

#: Dtypes the CPU memory model can represent exactly.
_NUMPY_DTYPES = {
    "bool": np.bool_,
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
    "index": np.int64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}

#: Dtypes the interpreter understands but refuses to execute.
_DEFERRED_DTYPES = {
    "bfloat16": "bfloat16 has no NumPy equivalent in the supported NumPy range.",
    "float8_e4m3fn": "float8 is out of scope for the first interpreter version.",
    "float8_e5m2": "float8 is out of scope for the first interpreter version.",
}

_INTEGER_KINDS = "iu"


def resolve_dtype(dtype):
    """Return the NumPy dtype for a NineToothed dtype name.

    :param dtype: The NineToothed dtype name, alias, or ``None``.
    :return: A ``numpy.dtype`` instance.
    :raises UnsupportedDTypeError: If the dtype cannot be represented exactly.
    """
    if dtype is None:
        raise UnsupportedDTypeError("Cannot resolve a missing dtype.")

    name = normalize_dtype(str(dtype))

    if name in _NUMPY_DTYPES:
        return np.dtype(_NUMPY_DTYPES[name])

    if name in _DEFERRED_DTYPES:
        reason = _DEFERRED_DTYPES[name].rstrip(".")

        raise UnsupportedDTypeError(f"Unsupported dtype `{name}`: {reason}.")

    raise UnsupportedDTypeError(f"Unknown dtype `{dtype}`.")


def is_supported_dtype(dtype) -> bool:
    """Return whether a NineToothed dtype name can be executed on the CPU."""
    try:
        resolve_dtype(dtype)
    except UnsupportedDTypeError:
        return False
    return True


def is_bool_dtype(dtype) -> bool:
    """Return whether a NumPy dtype is boolean."""
    return np.dtype(dtype).kind == "b"


def is_integer_dtype(dtype) -> bool:
    """Return whether a NumPy dtype is a signed or unsigned integer."""
    return np.dtype(dtype).kind in _INTEGER_KINDS


def identity(dtype, kind):
    """Return the neutral element used to seed a reduction accumulator.

    :param dtype: The NumPy dtype of the reduced value.
    :param kind: One of ``"sum"``, ``"max"`` or ``"min"``.
    :return: A NumPy scalar of ``dtype``.
    """
    dtype = np.dtype(dtype)

    if kind == "sum":
        return dtype.type(0)

    if kind == "max":
        if dtype.kind == "b":
            return dtype.type(False)

        if is_integer_dtype(dtype):
            return np.iinfo(dtype).min

        return dtype.type(-np.inf)

    if kind == "min":
        if dtype.kind == "b":
            return dtype.type(True)

        if is_integer_dtype(dtype):
            return np.iinfo(dtype).max

        return dtype.type(np.inf)

    raise UnsupportedDTypeError(f"Unknown reduction kind `{kind}`.")


def cast_array(array, dtype):
    """Cast ``array`` to ``dtype`` using backend-neutral semantics.

    ``bool`` casts follow the NumPy rule of ``value != 0``.  Integer casts wrap
    exactly like the C casts emitted by the CUDA backend.
    """
    dtype = np.dtype(dtype)

    if array.dtype == dtype:
        return array

    if dtype.kind == "b":
        return np.not_equal(array, 0)

    if is_integer_dtype(dtype):
        if array.dtype.kind == "f":
            truncated = np.trunc(array)

            return truncated.astype(dtype)

        return array.astype(dtype)

    return array.astype(dtype)


__all__ = [
    "cast_array",
    "identity",
    "is_bool_dtype",
    "is_integer_dtype",
    "is_supported_dtype",
    "resolve_dtype",
]
