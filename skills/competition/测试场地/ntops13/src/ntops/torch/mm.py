import torch

import ntops
from ntops.torch.utils import _cached_make, _get_matmul_input_precision


def mm(input, mat2, *, out=None):
    m, _ = input.shape
    _, n = mat2.shape

    if out is None:
        out = torch.empty((m, n), dtype=input.dtype, device=input.device)

    kernel = _cached_make(ntops.kernels.mm.premake)

    kernel(input, mat2, out, _get_matmul_input_precision())

    return out
