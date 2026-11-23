import torch

import ntops
from ntops.torch.utils import _cached_make


def div(input, other, *, rounding_mode=None, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.div.premake, input.ndim, rounding_mode)

    kernel(input, other, out)

    return out
