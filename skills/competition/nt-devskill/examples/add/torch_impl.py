import torch

from examples.add.kernel import kernel


def add(input, other):
    output = torch.empty_like(input)

    kernel(input, other, output, BLOCK_SIZE=1024)

    return output
