import torch

import ntops
from ntops.torch.utils import _cached_make


def logsumexp(input, dim, keepdim=False, *, out=None):
    tensor_dtype = out.dtype if out is not None else input.dtype

    output_shape = list(input.shape)
    output_shape[dim] = 1

    temp_out = torch.empty(output_shape, dtype=tensor_dtype, device=input.device)

    block_size = 256
    kernel = _cached_make(ntops.kernels.logsumexp.premake, input.ndim, dim, block_size)
    kernel(input, temp_out)

    if not keepdim:
        del output_shape[dim]
        temp_out = temp_out.view(output_shape)

    if out is not None:
        out.copy_(temp_out)
        return out

    return temp_out
