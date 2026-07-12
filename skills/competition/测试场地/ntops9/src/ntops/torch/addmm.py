import torch

import ntops
from ntops.torch.utils import _cached_make, _get_matmul_input_precision


def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    m, _ = mat1.shape
    _, n = mat2.shape

    if out is None:
        out = torch.empty((m, n), dtype=input.dtype, device=input.device)

    kernel = _cached_make(ntops.kernels.addmm.premake)

    kernel(input, mat1, mat2, beta, alpha, out, _get_matmul_input_precision())

    return out
