import torch

import ntops
from ntops.torch.utils import _cached_make


def bitwise_not(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(
        ntops.kernels.bitwise_not.premake, input.ndim, input.dtype == torch.bool
    )

    kernel(input, out)

    return out
