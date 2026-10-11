"""
T2: Tiled Softmax — follows official ninetoothed test_softmax.py pattern.

Key: ntl.max() and ntl.sum() reduce the ENTIRE tile to a scalar.
Tiling as (1, BLOCK_SIZE) = one row per tile → tile reduction = row reduction.
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


@ninetoothed.jit
def _softmax_kernel(
    input_row: Tensor(2, other=float("-inf")).tile((1, BLOCK_SIZE)),
    output_row: Tensor(2).tile((1, BLOCK_SIZE)),
):
    """Numerically stable row-wise softmax."""
    row_minus_max = input_row - ntl.max(input_row)
    numerator = ntl.exp(row_minus_max)
    denominator = ntl.sum(numerator)
    output_row = numerator / denominator  # noqa: F841


def tiled_softmax(X, Y):
    """Row-wise softmax with numerical stability.

    Args:
        X: torch.Tensor, shape (M, N), float32, on CUDA.
        Y: torch.Tensor, shape (M, N), float32, on CUDA (output, pre-allocated).
    """
    _softmax_kernel(X, Y, BLOCK_SIZE=X.shape[-1])
