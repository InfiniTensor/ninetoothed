import torch

import ntops
from ntops.torch.utils import _cached_make


def rot90(input, k=1, dims=(0, 1), *, out=None):
    if out is None:
        if k % 2 == 0:
            out = torch.empty_like(input)
        else:
            out_shape = list(input.shape)
            out_shape[dims[0]], out_shape[dims[1]] = (
                out_shape[dims[1]],
                out_shape[dims[0]],
            )
            out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    kernel = _cached_make(
        ntops.kernels.rot90.premake,
        input.ndim,
        k,
        tuple(dims),
        tuple(input.shape[i] for i in dims),
    )

    kernel(input, out)

    return out
