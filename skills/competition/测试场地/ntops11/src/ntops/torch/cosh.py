import torch

import ntops
from ntops.torch.utils import _cached_make


def cosh(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.cosh.premake, input.ndim)

    kernel(input, out)

    return out
