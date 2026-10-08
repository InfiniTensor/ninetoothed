"""Example 2 — parameterized row reduction (reduction = none | mean | sum).

Family: reduction / blocking.
Confidence: MEDIUM. The row-tile + fp32-accumulate pattern follows the verified
softmax / rms_norm kernels. The one part to validate first on a CUDA host is the
scalar -> (1,1) output assignment in `application`. If it does not compile,
switch to the axis-reduction form noted below (kept as `application_axis`).

reduction='none' -> identity (pure elementwise; high confidence)
reduction='sum'  -> row sum  (B,N)->(B,)
reduction='mean' -> row mean (B,N)->(B,)
"""

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    # Input (B, N) tiled per row; output (B, 1) one scalar per row.
    return input.tile((1, BLOCK_SIZE)), output.tile((1, 1))


def application(input, output):
    # Fp32 accumulation is the main correctness guard for fp16/bf16 inputs.
    output = ntl.sum(ntl.cast(input, ntl.float32))  # noqa: F841


# Alternative if the scalar->(1,1) assignment above fails to compile:
#   def application_axis(input, output):
#       output = ntl.sum(ntl.cast(input, ntl.float32), axis=1)[:, None]  # noqa: F841
# Uncomment the axis variant above if the scalar assignment fails to compile.

_sum_kernel = ninetoothed.make(
    arrangement, application, (Tensor(2, other=0.0), Tensor(2))
)


def row_reduce(x, reduction="mean"):
    """x: (B, N). reduction in {'none','mean','sum'}."""
    if reduction == "none":
        return x.clone()

    b, n = x.shape
    out = torch.empty(b, 1, dtype=x.dtype, device=x.device)
    _sum_kernel(x, out, BLOCK_SIZE=n)
    s = out.squeeze(-1)  # (B,).

    return s / n if reduction == "mean" else s


def reference(x, reduction="mean"):
    if reduction == "none":
        return x.clone()

    if reduction == "sum":
        return x.sum(dim=-1)
    return x.mean(dim=-1)
