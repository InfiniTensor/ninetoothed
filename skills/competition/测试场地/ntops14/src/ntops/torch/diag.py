import builtins

import torch

import ntops
from ntops.torch.utils import _cached_make


def diag(input, diagonal=0):
    if input.ndim == 1:
        return _diag_embed(input, diagonal)
    elif input.ndim == 2:
        return _diag_extract(input, diagonal)
    else:
        raise ValueError(f"Input must be 1-D or 2-D, but got {input.ndim}-D.")


def _diag_embed(input, diagonal):
    n = input.shape[0]
    size = n + builtins.abs(diagonal)
    output = torch.zeros((size, size), dtype=input.dtype, device=input.device)

    if n == 0:
        return output

    output_flat = output.view(-1)

    if diagonal >= 0:
        start = diagonal
    else:
        start = (-diagonal) * size

    stride = size + 1

    kernel = _cached_make(ntops.kernels.diag.premake_embed, stride=stride)
    kernel(input, output_flat[start:])

    return output


def _diag_extract(input, diagonal):
    m, n = input.shape

    if diagonal >= 0:
        diag_len = max(min(m, n - diagonal), 0)
        start = diagonal
    else:
        diag_len = max(min(m + diagonal, n), 0)
        start = (-diagonal) * n

    output = torch.empty(diag_len, dtype=input.dtype, device=input.device)

    if diag_len == 0:
        return output

    input_flat = input.contiguous().view(-1)
    stride = n + 1

    kernel = _cached_make(ntops.kernels.diag.premake_extract, stride=stride)
    kernel(input_flat[start:], output)

    return output
