import random

import torch

import ntops
from ntops.torch.utils import _cached_make


def dropout(input, p=0.5, training=True, inplace=False):
    if not training or p == 0:
        if inplace:
            return input
        else:
            return input.clone()

    seed = random.randrange(0, 2**31)

    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.dropout.premake, input.ndim)

    kernel(input, p, seed, output)

    return output
