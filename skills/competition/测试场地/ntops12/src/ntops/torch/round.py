import torch

import ntops
from ntops.torch.utils import _cached_make


def round(input, decimals=0, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    if decimals == 0:
        kernel = _cached_make(ntops.kernels.round.premake, input.ndim)
        kernel(input, out)
    else:
        factor = 10.0**decimals
        inv_factor = 1.0 / factor
        kernel = _cached_make(ntops.kernels.round.premake, input.ndim, decimals=True)
        kernel(input, factor, inv_factor, out)

    return out
