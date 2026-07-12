"""T4 self-test — L4 performance / diagnosis / integration lane.

This script is BOTH the benchmark (Part A) and a runnable performance-regression
diagnosis (Part B). It measures the elementwise multiply kernel two ways:

- ``naive``: rebuilds the ``@ninetoothed.jit`` kernel on every call (a common
  mistake), so each timed call pays recompilation/auto-tuning.
- ``fixed``: builds the kernel once and reuses it, so timing reflects real compute.

The gap between the two is the regression the skill's failure table (row #12) tells
the agent to hunt down. Baseline is native ``torch`` elementwise ``*``.

Run from the repository root:
    python skills/competition/ninetoothed-operator-skill/examples/t4-performance-diagnosis/benchmark_multiply.py
"""

import pathlib
import sys

import torch

import ninetoothed
from ninetoothed import Symbol, Tensor

_SKILL_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SKILL_ROOT / "scripts"))

from bench import compare  # noqa: E402


def _build_kernel():
    """Build the multiply kernel once. Reused across calls to avoid recompiling."""
    BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)

    @ninetoothed.jit
    def multiply_kernel(
        lhs: Tensor(1).tile((BLOCK_SIZE,)),
        rhs: Tensor(1).tile((BLOCK_SIZE,)),
        output: Tensor(1).tile((BLOCK_SIZE,)),
    ):
        output = lhs * rhs  # noqa: F841

    return multiply_kernel


# The fixed operator builds its kernel exactly once, at import time.
_MULTIPLY_KERNEL = _build_kernel()


def multiply_fixed(lhs, rhs):
    output = torch.empty_like(lhs)

    _MULTIPLY_KERNEL(lhs, rhs, output)

    return output


def multiply_naive(lhs, rhs):
    # Anti-pattern: rebuilds and recompiles the kernel on every single call.
    kernel = _build_kernel()
    output = torch.empty_like(lhs)

    kernel(lhs, rhs, output)

    return output


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for size in (1 << 20, 1 << 24):
        lhs = torch.rand(size, dtype=torch.float32, device=device)
        rhs = torch.rand(size, dtype=torch.float32, device=device)

        # Warm up the compile + auto-tune for the fixed kernel before timing.
        multiply_fixed(lhs, rhs)

        print(f"\n=== size = {size} ===")
        compare(
            candidate=lambda: multiply_fixed(lhs, rhs),
            baseline=lambda: lhs * rhs,
            label=f"multiply_fixed[{size}]",
            input_desc=f"shape=({size},), dtype=float32, layout=contiguous",
        )


if __name__ == "__main__":
    main()
