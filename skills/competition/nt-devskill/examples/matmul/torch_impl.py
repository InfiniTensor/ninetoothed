import torch

from examples.matmul.kernel import kernel


def mm(input, other):
    output_shape = (input.shape[0], other.shape[1])
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    kernel(input, other, output)

    return output


def matmul(input, other):
    return mm(input, other)
