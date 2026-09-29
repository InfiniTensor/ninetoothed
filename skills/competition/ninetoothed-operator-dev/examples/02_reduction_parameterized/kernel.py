"""
Row-wise sum reduction kernel (reduces last dim).

reduction='none'  → no kernel needed; wrapper returns x (or x**2 for MSE demo)
reduction='sum'   → this kernel; wrapper returns the result directly
reduction='mean'  → this kernel; wrapper divides by N after the kernel

Design notes
------------
* tile((1, BLOCK_SIZE)): one block tile = one row of the input.
  The application reduces the entire tile → one scalar per row.
* fp32 accumulate: even fp16 inputs accumulate in float32 to avoid overflow.
  The cast happens inside application; the output dtype mirrors the input.
* other=0.0 on the input Tensor: OOB lanes contribute 0 to the sum (correct
  for sum and mean reductions over partial rows).
* BLOCK_SIZE is passed as the column count N (covering one full row per tile).
  For very large N, split into sub-blocks and accumulate; this kernel handles
  up to BLOCK_SIZE = N columns directly.
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(x, out, BLOCK_SIZE=BLOCK_SIZE):
    """Tile each row into one program and each output into one scalar.

    X  : (M, N)   → tile((1, BLOCK_SIZE)) → one row per program.
    out: (M,)     → tile((1,))            → one scalar per program.
    """
    return x.tile((1, BLOCK_SIZE)), out.tile((1,))


def application(x, out):
    # Cast to fp32 for numerically stable accumulation.
    x_fp32 = ntl.cast(x, ntl.float32)
    total = ntl.sum(x_fp32)
    out = ntl.cast(total, out.dtype)  # noqa: F841


_TENSORS = (Tensor(2, other=0.0), Tensor(1))

kernel = ninetoothed.make(arrangement, application, _TENSORS)
