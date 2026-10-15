"""Example 4 — softmax performance-regression diagnosis.

Family: performance / diagnosis.
Confidence: HIGH — kernel follows the `softmax` pattern from the NineToothed
repo's own `tests/test_softmax.py`. Two wrappers differ only in the block size:

  softmax_fast  -> BLOCK_SIZE = row length N            (no wasted lanes)
  softmax_slow  -> BLOCK_SIZE = SLOW_BLOCK (>> N)        (mostly-masked lanes)

Both are numerically correct: out-of-range lanes read `other=-inf`, so
exp(-inf)=0 and the max is unchanged. `softmax_slow` runs exp/max/sum over the
extra masked lanes — masked loads fetch no DRAM, so the waste is vector
compute/occupancy, not bandwidth. The regression is real and reproducible but
ratio-dependent: ~1.6-1.9x slower when SLOW_BLOCK >> N (N <= 1024), shrinking to
parity as N approaches the block size and the kernel becomes bandwidth-bound.
Diagnose with inspect_generated_source.py (deterministic block-size diff) +
bench_compare.py (wall-clock at small N).
"""

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
SLOW_BLOCK = 8192  # Deliberately oversized to waste masked lanes.


def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((1, BLOCK_SIZE)), output.tile((1, BLOCK_SIZE))


def application(input, output):
    row_minus_max = input - ntl.max(input)
    numerator = ntl.exp(row_minus_max)
    output = numerator / ntl.sum(numerator)  # noqa: F841


_kernel = ninetoothed.make(
    arrangement, application, (Tensor(2, other=float("-inf")), Tensor(2))
)


def softmax_fast(x):
    out = torch.empty_like(x)
    _kernel(x, out, BLOCK_SIZE=x.shape[-1])  # Exact row length.

    return out


def softmax_slow(x):
    out = torch.empty_like(x)
    _kernel(x, out, BLOCK_SIZE=SLOW_BLOCK)  # Oversized -> wasted lanes.

    return out


def reference(x):
    return torch.softmax(x, dim=-1)
