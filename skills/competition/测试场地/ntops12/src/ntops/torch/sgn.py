import torch

import ntops
from ntops.torch.utils import _cached_make


def sgn(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    if input.dtype not in (torch.complex64, torch.complex128):
        kernel = _cached_make(ntops.kernels.sign.premake, input.ndim)
        kernel(input, out)
    else:
        input = torch.view_as_real(input)
        out_rm = torch.view_as_real(out)

        kernel = _cached_make(ntops.kernels.sgn.premake, input.ndim)
        kernel(input, out_rm)

    return out
