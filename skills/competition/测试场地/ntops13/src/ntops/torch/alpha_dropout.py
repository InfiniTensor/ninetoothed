import math
import random

import torch

import ntops
from ntops.torch.utils import _cached_make

# SELU saturation value: -lambda * alpha
_ALPHA_P = -1.7580993408473766


def alpha_dropout(input, p=0.5, training=False, inplace=False):
    if not training or p == 0:
        if inplace:
            return input
        else:
            return input.clone()

    q = 1.0 - p
    a = 1.0 / math.sqrt(q * (1.0 + p * _ALPHA_P * _ALPHA_P))
    b = -a * p * _ALPHA_P
    sat = a * _ALPHA_P + b

    seed = random.randrange(0, 2**31)

    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.alpha_dropout.premake, input.ndim)

    kernel(input, a, b, sat, p, seed, output)

    return output
