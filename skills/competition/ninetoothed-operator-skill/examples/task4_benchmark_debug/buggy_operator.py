"""
T4 Sub-Task 4C: Buggy Softmax — uses WRONG API patterns.

This version deliberately uses the initial (incorrect) API assumptions:
- ntl.max() with dim= and keepdims= (not supported in NineToothed 0.26)
- ntl.sum() with dim= and keepdims= (not supported)

Expected: CompilationError — Triton cannot compile this kernel.

This reproduces the REAL debugging experience encountered during
development of T2. The fix is documented in ../task2_reduction_block/operator_impl.py
and the diagnosis in diagnosis_record.md.
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_M = Symbol("BLOCK_M", meta=True)
BLOCK_N = Symbol("BLOCK_N", meta=True)


@ninetoothed.jit
def _buggy_softmax_kernel(
    input_row: Tensor(2, other=float("-inf")).tile((BLOCK_M, BLOCK_N)),
    output_row: Tensor(2).tile((BLOCK_M, BLOCK_N)),
):
    """WRONG: dim=/keepdims= not supported in NineToothed 0.26."""
    x_max = ntl.max(input_row, dim=1, keepdims=True)
    x_shifted = input_row - x_max
    exp_x = ntl.exp(x_shifted)
    exp_sum = ntl.sum(exp_x, dim=1, keepdims=True)
    output_row = exp_x / exp_sum  # noqa: F841


def tiled_softmax_buggy(X, Y):
    """BUGGY — will raise CompilationError."""
    _buggy_softmax_kernel(X, Y)
