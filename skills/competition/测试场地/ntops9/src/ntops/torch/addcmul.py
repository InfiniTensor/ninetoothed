import torch

import ntops
from ntops.torch.utils import _cached_make


def addcmul(input, tensor1, tensor2, *, value=1, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.addcmul.premake, input.ndim)

    kernel(input, tensor1, tensor2, value, out)

    return out
