import torch

import ntops
from ntops.torch.utils import _cached_make


def add(input, other, *, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.add.premake, input.ndim)

    kernel(input, other, alpha, out)

    return out
