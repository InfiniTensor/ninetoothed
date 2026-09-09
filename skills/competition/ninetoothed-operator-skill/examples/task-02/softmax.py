import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor


def softmax(input):
    """Compute row-wise softmax over the last dimension of a 2D CUDA tensor."""
    if input.dim() != 2:
        raise NotImplementedError(
            f"softmax expects a 2D tensor, got shape={tuple(input.shape)}"
        )

    block = Symbol("BLOCK_SIZE", constexpr=True)

    @ninetoothed.jit
    def softmax_kernel(
        input_row: Tensor(2, other=float("-inf")).tile((1, block)),
        output_row: Tensor(2).tile((1, block)),
    ):
        row_minus_max = input_row - ntl.max(input_row)
        numerator = ntl.exp(row_minus_max)
        denominator = ntl.sum(numerator)
        output_row = numerator / denominator  # noqa: F841

    output = torch.empty_like(input)
    softmax_kernel(input, output, BLOCK_SIZE=input.shape[-1])
    return output
