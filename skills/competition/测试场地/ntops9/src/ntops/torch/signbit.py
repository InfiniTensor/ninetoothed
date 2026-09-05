import torch

import ntops
from ntops.torch.utils import _cached_make


def signbit(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)

    kernel = _cached_make(ntops.kernels.signbit.premake, input.ndim)

    kernel(input, out)

    return out
