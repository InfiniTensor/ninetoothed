import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def _next_power_of_two(value):
    if value <= 1:
        return 1

    return 1 << (value - 1).bit_length()


def application(input, values, indices, sort_size, descending):
    input_0 = input[0]
    offsets = ntl.arange(0, input_0.shape[0])
    valid = offsets < sort_size

    sign_mask = ntl.cast(0x7FFFFFFF, ntl.int32)
    input_fp32 = ntl.cast(input_0, ntl.float32)
    encoded = ntl.cast(input_fp32, ntl.int32, bitcast=True)
    encoded = encoded ^ ((encoded >> 31) & sign_mask)

    if descending:
        encoded = ~encoded

    encoded = ntl.where(valid, encoded, ntl.cast(0x7FFFFFFF, ntl.int32))

    offsets = ntl.cast(offsets, ntl.int64)
    key = ((ntl.cast(encoded, ntl.int64) & ntl.cast(0xFFFFFFFF, ntl.int64)) << 32) | offsets
    sorted_key = ntl.sort(key)

    sorted_encoded = ntl.cast(sorted_key >> 32, ntl.int32)

    if descending:
        sorted_encoded = ~sorted_encoded

    sorted_encoded = sorted_encoded ^ ((sorted_encoded >> 31) & sign_mask)
    sorted_values = ntl.cast(sorted_encoded, ntl.float32, bitcast=True)
    sorted_indices = sorted_key & ntl.cast(0xFFFFFFFF, ntl.int64)

    values[0] = ntl.cast(sorted_values, values[0].dtype)
    indices[0] = ntl.cast(sorted_indices, indices[0].dtype)


def premake(
    ndim,
    dim,
    sort_size,
    descending=False,
    stable=False,
    dtype=None,
    block_size=None,
):
    if block_size is None:
        block_size = _next_power_of_two(sort_size)

    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    # `stable` is kept for `torch.sort` interface parity. Current key design is stable.
    _ = stable

    tensors = (
        Tensor(ndim, dtype=dtype, other=0),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=ninetoothed.int64),
        Tensor(0, constexpr=True, value=sort_size),
        Tensor(0, constexpr=True, value=descending),
    )

    return arrangement_, application, tensors
