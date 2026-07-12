import torch

import ntops
from ntops.torch.utils import _cached_make


def tanh(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.tanh.premake, input.ndim)

    kernel(input, out)

    return out
