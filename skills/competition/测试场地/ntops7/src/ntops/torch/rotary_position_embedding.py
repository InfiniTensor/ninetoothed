import torch

import ntops
from ntops.torch.utils import _cached_make


def rotary_position_embedding(
    input, sin_table, cos_table, interleaved=True, inplace=False
):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    batch_size, _, num_heads, _ = input.shape

    sin_table = sin_table[None, :, None, :].expand(batch_size, -1, num_heads, -1)
    cos_table = cos_table[None, :, None, :].expand(batch_size, -1, num_heads, -1)

    kernel = _cached_make(
        ntops.kernels.rotary_position_embedding.premake,
        input.ndim,
        interleaved=interleaved,
        num_warps=1,
    )

    kernel(input, sin_table, cos_table, output)

    return output
