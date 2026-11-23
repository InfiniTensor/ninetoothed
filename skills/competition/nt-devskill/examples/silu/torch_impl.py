import torch

from examples.silu.kernel import kernel


def silu(input):
    input_flat = input.flatten()
    output_flat = torch.empty_like(input_flat)

    kernel(input_flat, output_flat, BLOCK_SIZE=1024)

    return output_flat.view_as(input)
