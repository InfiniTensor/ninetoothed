import torch

import ntops
from ntops.torch.utils import _cached_make, _get_matmul_input_precision


def bmm(input, mat2, *, out=None):
    b, m, _ = input.shape
    _, _, n = mat2.shape

    if out is None:
        out = torch.empty((b, m, n), dtype=input.dtype, device=input.device)

    kernel = _cached_make(ntops.kernels.bmm.premake)

    kernel(input, mat2, out, _get_matmul_input_precision())

    return out
