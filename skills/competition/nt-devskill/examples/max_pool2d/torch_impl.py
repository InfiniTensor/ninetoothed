import math

import torch

from examples.max_pool2d.kernel import kernel


def max_pool2d(input, window_shape):
    n, c, h, w = input.shape
    r, s = window_shape
    p = math.ceil((h - r) / r + 1)
    q = math.ceil((w - s) / s + 1)

    output = torch.empty(n, c, p, q, dtype=input.dtype, device=input.device)

    kernel(input, output, WINDOW_HEIGHT=r, WINDOW_WIDTH=s)

    return output
