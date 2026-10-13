import torch

import ntops
from ntops.torch.utils import _cached_make


def exp(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.exp.premake, input.ndim)

    kernel(input, out)

    return out
