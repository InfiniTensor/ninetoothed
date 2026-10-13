import torch

from examples.swiglu.kernel import kernel


def swiglu(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()

    c = torch.empty_like(a_flat)

    kernel(a_flat, b_flat, c, BLOCK_SIZE=1024)

    return c.view_as(a)
