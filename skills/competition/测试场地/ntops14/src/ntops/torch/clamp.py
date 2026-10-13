import torch

import ntops
from ntops.torch.utils import _cached_make


def clamp(input, min=None, max=None, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.clamp.premake, input.ndim)

    kernel(input, min, max, out)

    return out
