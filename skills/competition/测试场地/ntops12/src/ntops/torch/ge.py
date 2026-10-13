import torch

import ntops
from ntops.torch.utils import _cached_make


def ge(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.ge.premake, input.ndim)

    kernel(input, other, out)

    return out
