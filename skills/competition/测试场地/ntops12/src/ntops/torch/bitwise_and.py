import torch

import ntops
from ntops.torch.utils import _cached_make


def bitwise_and(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.bitwise_and.premake, input.ndim)

    kernel(input, other, out)

    return out
