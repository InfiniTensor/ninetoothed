import torch

from examples.rotary_position_embedding.kernel import kernel


def rotary_position_embedding(input, sin_table, cos_table, interleaved=True):
    batch_size, _, num_heads, _ = input.shape

    output = input.clone()
    sin_table = sin_table[None, :, None, :].expand(batch_size, -1, num_heads, -1)
    cos_table = cos_table[None, :, None, :].expand(batch_size, -1, num_heads, -1)

    kernel(output, sin_table, cos_table, interleaved)

    return output
