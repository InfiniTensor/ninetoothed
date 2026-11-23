import torch

import ntops
from ntops.torch.utils import _cached_make


def pow(input, exponent, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.pow.premake, input.ndim)

    kernel(input, exponent, out)

    return out
