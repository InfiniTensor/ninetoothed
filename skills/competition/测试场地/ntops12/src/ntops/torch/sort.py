import torch

import ntops
from ntops.torch.utils import _cached_make


def sort(input, dim=-1, descending=False, stable=False, *, out=None):
    assert input.device.type == "cuda", "`input` must be on CUDA."
    assert input.ndim > 0, "`input` must have at least one dimension."
    assert input.dtype in (torch.float16, torch.bfloat16, torch.float32), (
        "`input.dtype` must be one of float16, bfloat16, or float32."
    )

    if dim < 0:
        dim += input.ndim

    if dim < 0 or dim >= input.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-input.ndim}, {input.ndim - 1}], but got {dim})"
        )

    sort_size = input.shape[dim]

    assert sort_size > 0, "`input.shape[dim]` must be greater than 0."

    if out is None:
        values = torch.empty_like(input)
        indices = torch.empty_like(input, dtype=torch.int64)
    else:
        values, indices = out

    kernel = _cached_make(
        ntops.kernels.sort.premake,
        input.ndim,
        dim,
        sort_size=sort_size,
        descending=descending,
    )

    kernel(input, values, indices, sort_size, descending)

    return torch.return_types.sort((values, indices))
