import torch

import ntops
from ntops.torch.utils import _cached_make


def threshold(input, threshold, value, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.threshold.premake, input.ndim)

    kernel(input, threshold, value, output)

    return output
