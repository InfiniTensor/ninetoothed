import torch

from examples.bmm.kernel import kernel


def bmm(lhs, rhs):
    output_shape = (lhs.shape[0], lhs.shape[-2], rhs.shape[-1])
    output = torch.empty(output_shape, dtype=lhs.dtype, device=lhs.device)

    kernel(lhs, rhs, output)

    return output
