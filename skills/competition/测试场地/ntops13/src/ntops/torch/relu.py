import torch

import ntops
from ntops.torch.utils import _cached_make


def relu(input, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.relu.premake, input.ndim)

    kernel(input, output)

    return output
