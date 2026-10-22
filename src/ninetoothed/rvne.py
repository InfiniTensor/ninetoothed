"""Explicit packed data conversion for the RVNE spike accumulator."""

import numpy as np


def pack_spikes(values):
    """Pack the last axis into uint32 words, with the first spike in bit zero.

    Values must already be binary. The final word is padded with zero spikes;
    this function never thresholds or quantizes an input tensor.
    """
    values = np.asarray(values)
    _validate_integer_array(values)

    if np.any((values != 0) & (values != 1)):
        raise ValueError("RVNE spikes must contain only zero and one.")

    return _pack(values, lanes=32, bits=1, dtype=np.uint32)


def pack_weights(values):
    """Pack signed INT4 weights into uint64 words, lowest nibble first.

    Each word holds sixteen two's-complement weights. The final word is padded
    with zero weights. Inputs must be integers in [-8, 7]; quantization belongs
    to the caller and is never performed implicitly.
    """
    values = np.asarray(values)
    _validate_integer_array(values)

    if np.any(values < -8) or np.any(values > 7):
        raise ValueError("RVNE weights must be signed INT4 integers in [-8, 7].")

    return _pack(values, lanes=16, bits=4, dtype=np.uint64)


def _validate_integer_array(values):
    if values.ndim == 0:
        raise ValueError("RVNE packing requires an array with at least one axis.")

    if values.dtype.kind not in "biu":
        raise TypeError("RVNE packing requires integer or boolean input values.")


def _pack(values, *, lanes, bits, dtype):
    count = values.shape[-1]
    words = (count + lanes - 1) // lanes
    padded = np.zeros((*values.shape[:-1], words * lanes), dtype=dtype)
    padded[..., :count] = values.astype(dtype) & dtype((1 << bits) - 1)
    padded = padded.reshape((*values.shape[:-1], words, lanes))
    shifts = np.arange(lanes, dtype=dtype) * dtype(bits)

    return np.bitwise_or.reduce(padded << shifts, axis=-1)
