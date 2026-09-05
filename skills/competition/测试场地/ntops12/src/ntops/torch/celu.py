import torch

import ntops
from ntops.torch.utils import _cached_make


def celu(input, alpha=1.0, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.celu.premake, input.ndim)

    kernel(input, alpha, output)

    return output
