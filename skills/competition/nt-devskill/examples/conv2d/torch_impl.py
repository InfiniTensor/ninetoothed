import torch

from examples.conv2d.kernel import kernel


def conv2d(input, filter):
    n, _, h, w = input.shape
    k, _, r, s = filter.shape
    p = h - r + 1
    q = w - s + 1

    output = torch.empty((n, k, p, q), dtype=input.dtype, device=input.device)

    kernel(input, filter, output)

    return output
