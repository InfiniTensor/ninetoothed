import torch

import ntops
from ntops.torch.utils import _cached_make


def softmax(input, dim, dtype=None):
    tensor_dtype = dtype if dtype is not None else input.dtype

    output = torch.empty_like(input, dtype=tensor_dtype)

    kernel = _cached_make(ntops.kernels.softmax.premake, input.ndim, dim)

    kernel(input, output)

    return output
