import math

import torch

from examples.scaled_dot_product_attention.kernel import kernel


def scaled_dot_product_attention(q, k, v, scale=None):
    if scale is None:
        scale = 1 / math.sqrt(q.shape[-1])

    o = torch.empty_like(q)

    kernel(q, k, v, scale, o)

    return o
