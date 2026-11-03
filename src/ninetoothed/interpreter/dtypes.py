"""NumPy dtype mapping for the CPU reference interpreter."""

import numpy as np

from ninetoothed.interpreter.errors import UnsupportedDTypeError

_DTYPE_ALIASES = {
    "bool": "bool",
    "i1": "bool",
    "int1": "bool",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "float8": "float8",
    "float8e4m3fn": "float8",
    "float8e5m2": "float8",
    "float": "float32",
    "fp16": "float16",
    "fp32": "float32",
    "fp64": "float64",
}

_NUMPY_DTYPES = {
    "bool": np.bool_,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
}

_INDEX_DTYPE = np.dtype(np.int64)

_UNSUPPORTED_DTYPES = {
    "bfloat16": "NumPy has no native `bfloat16`, and the interpreter does not "
    "emulate it",
    "float8": "`float8` arithmetic is not defined for the CPU interpreter",
}


def normalize_dtype_name(dtype) -> str | None:
    """Return the canonical dtype name for an SSA dtype spelling.

    :param dtype: The dtype spelling recorded in the SSA, for example
        ``ntl.float32`` or ``tl.int32``.
    :return: The canonical dtype name, or ``None`` when no dtype is recorded.
    """
    if dtype is None:
        return None

    text = str(dtype).strip().strip("'\"")

    if "." in text:
        text = text.split(".")[-1]

    if not text:
        return None

    return _DTYPE_ALIASES.get(text, text)


def resolve_dtype(dtype) -> np.dtype | None:
    """Return the NumPy dtype used to execute an SSA dtype.

    :param dtype: The dtype spelling recorded in the SSA.
    :return: The NumPy dtype, or ``None`` when no dtype is recorded.
    :raises UnsupportedDTypeError: When the dtype has no NumPy equivalent.
    """
    name = normalize_dtype_name(dtype)

    if name is None:
        return None

    if name in _UNSUPPORTED_DTYPES:
        raise UnsupportedDTypeError(
            f"Unsupported dtype `{name}` for the CPU interpreter: "
            f"{_UNSUPPORTED_DTYPES[name]}."
        )

    try:
        return np.dtype(_NUMPY_DTYPES[name])
    except KeyError as exc:
        raise UnsupportedDTypeError(
            f"Unsupported dtype `{name}` for the CPU interpreter."
        ) from exc


def is_float(dtype) -> bool:
    """Return whether a NumPy dtype holds floating-point values."""
    return dtype is not None and np.issubdtype(np.dtype(dtype), np.floating)


def is_bool(dtype) -> bool:
    """Return whether a NumPy dtype holds booleans."""
    return dtype is not None and np.dtype(dtype) == np.dtype(np.bool_)


def is_integer(dtype) -> bool:
    """Return whether a NumPy dtype holds integers."""
    return dtype is not None and np.issubdtype(np.dtype(dtype), np.integer)


def dtype_of(value) -> np.dtype | None:
    """Return the NumPy dtype of an interpreter value, if it has one."""
    if isinstance(value, np.ndarray):
        return value.dtype

    if isinstance(value, (bool, np.bool_)):
        return np.dtype(np.bool_)

    if isinstance(value, (int, np.integer)):
        return np.dtype(np.int64)

    if isinstance(value, (float, np.floating)):
        return np.dtype(np.float64)

    return None


__all__ = [
    "dtype_of",
    "is_bool",
    "is_float",
    "is_integer",
    "normalize_dtype_name",
    "resolve_dtype",
]
