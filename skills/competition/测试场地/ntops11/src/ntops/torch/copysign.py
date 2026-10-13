import torch

import ntops
from ntops.torch.utils import _cached_make


_BLOCK_SIZE = 4096


def copysign(input, other, *, out=None):
    assert input.shape == other.shape, (
        f"copysign requires input and other to have the same shape, "
        f"got {input.shape} and {other.shape}"
    )
    assert input.dtype == other.dtype, (
        f"copysign requires input and other to have the same dtype, "
        f"got {input.dtype} and {other.dtype}"
    )

    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(
        ntops.kernels.copysign.premake,
        input.ndim,
        input.dtype,
        _BLOCK_SIZE,
    )

    kernel(input, other, out)

    return out

