"""
FIXED softmax kernel — the 'after' in the perf/diag self-test task.

Fixes applied (both detectable in generated Triton source via
inspect_generated_source.py):

Fix 1 — subtract row max before exp:
  row_minus_max = x - ntl.max(x)   # stable; max never overflows exp
  This adds one ntl.max call, visible in the generated source as an extra
  tl.reduce operation.

Fix 2 — accumulate in fp32:
  ntl.cast(x, ntl.float32) before sum
  This adds tl.to(fp32) and tl.to(input_dtype) calls in the Triton source.

Both fixes match the canonical softmax.py in the ninetoothed-examples repo.
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(x, out, BLOCK_SIZE=BLOCK_SIZE):
    return x.tile((1, BLOCK_SIZE)), out.tile((1, BLOCK_SIZE))


def application(x, out):
    # FIX 1: subtract row max for numerical stability.
    row_minus_max = x - ntl.max(x)
    # FIX 2: accumulate in fp32.
    numerator = ntl.exp(ntl.cast(row_minus_max, ntl.float32))
    out = ntl.cast(numerator / ntl.sum(numerator), out.dtype)  # noqa: F841


_TENSORS = (Tensor(2, other=float("-inf")), Tensor(2))

kernel_fixed = ninetoothed.make(arrangement, application, _TENSORS)
