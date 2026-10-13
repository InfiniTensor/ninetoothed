import torch

import ntops
from ntops.torch.utils import _cached_make


def select_copy(input, dim, index, *, out=None):
    if out is None:
        shape = tuple(input.size(i) for i in range(input.ndim) if i != dim)
        out = torch.empty(shape, dtype=input.dtype, device=input.device)

    kernel = _cached_make(ntops.kernels.select_copy.premake, input.ndim, out.ndim, dim)

    kernel(input, index, out)

    return out
