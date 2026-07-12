import math

import torch

import ntops
from ntops.torch.utils import _cached_make


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    normalized_shape = tuple(normalized_shape)

    if weight is None:
        weight = torch.ones_like(input)
    else:
        weight = weight.expand_as(input)

    if bias is None:
        bias = torch.zeros_like(input)
    else:
        bias = bias.expand_as(input)

    output = torch.empty_like(input)

    kernel = _cached_make(
        ntops.kernels.layer_norm.premake, input.ndim, normalized_shape
    )

    kernel(input, weight, bias, eps, output, math.prod(normalized_shape))

    return output
