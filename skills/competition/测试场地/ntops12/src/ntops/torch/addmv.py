import torch

import ntops
from ntops.torch.utils import _cached_make, _get_matmul_input_precision


def addmv(input, mat, vec, *, beta=1, alpha=1, out=None):
    m, k = mat.shape
    
    if vec.shape[0] != k:
        raise RuntimeError(f"size mismatch, got {vec.shape[0]}, expected {k}")

    if out is None:
        out = torch.empty((m,), dtype=input.dtype, device=input.device)

    # 准备 view: (N,) -> (N, 1)
    # 这样底层 kernel 看到的是矩阵，从而复用 mm 逻辑
    input_2d = input.unsqueeze(1)
    vec_2d = vec.unsqueeze(1)
    out_2d = out.unsqueeze(1)

    kernel = _cached_make(ntops.kernels.addmv.premake)

    # 传入 2D view
    kernel(input_2d, mat, vec_2d, beta, alpha, out_2d, _get_matmul_input_precision())

    return out