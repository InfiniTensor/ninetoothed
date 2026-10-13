import torch

from examples.softmax.kernel import kernel


def softmax(input):
    output = torch.empty_like(input)

    kernel(input, output, BLOCK_SIZE=input.shape[-1])

    return output
