import torch

import ntops
from ntops.torch.utils import _cached_make


def quantile(input, q, dim=None, keepdim=False, interpolation="linear", out=None):
    if isinstance(q, float):
        q = torch.tensor(q, dtype=input.dtype, device=input.device)

    if out is None:
        if dim is not None:
            out_shape = list(input.shape)
            out_shape[dim] = 1
        else:
            out_shape = [1] * input.ndim

        if not keepdim:
            if dim is not None:
                out_shape.pop(dim)
            else:
                out_shape = []

        if q.ndim > 0:
            out_shape.insert(0, q.shape[0])

        print("Output shape:", out_shape)
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    if q.ndim == 0:
        # `triton.language.gather` does not support 0-dim tensors.
        q = q.unsqueeze(0)
        out_adjusted = out.unsqueeze(0)
    else:
        out_adjusted = out

    kernel = _cached_make(
        ntops.kernels.quantile.premake,
        input.ndim,
        out_adjusted.ndim,
        dim,
        interpolation,
    )

    kernel(input, q, out_adjusted)

    return out
