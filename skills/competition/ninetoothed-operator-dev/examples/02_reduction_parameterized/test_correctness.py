"""
Correctness + benchmark tests for parameterized reduction.

At least 2 self-test tasks require a benchmark; this is one of them.
"""

import itertools
import pathlib as _pathlib
import sys as _sys

import pytest
import torch

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
from wrapper import reduce_last_dim

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"
TOL = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),  # Fp32 sum: slight accumulation variance
    torch.float16: dict(atol=1e-2, rtol=1e-2),  # fp16 input, fp32 internal.
}
REDUCTIONS = ["none", "sum", "mean"]
SHAPES = [(32, 128), (64, 511), (128, 1024)]  # Incl. non-pow2.
DTYPES = [torch.float32, torch.float16]


def reference(x, reduction):
    if reduction == "none":
        return x.clone()

    if reduction == "sum":
        return x.sum(dim=-1)
    return x.mean(dim=-1)


@pytest.mark.parametrize(
    "shape,dtype,reduction",
    list(itertools.product(SHAPES, DTYPES, REDUCTIONS)),
)
def test_reduce_last_dim(shape, dtype, reduction):
    x = torch.randn(*shape, dtype=dtype, device=DEVICE)
    expected = reference(x, reduction)
    got = reduce_last_dim(x, reduction)
    torch.testing.assert_close(got, expected, **TOL[dtype])


def test_nan_inf_boundary():
    """Sum of a row containing NaN propagates NaN (expected behaviour)."""
    x = torch.ones(4, 16, dtype=torch.float32, device=DEVICE)
    x[0, 0] = float("nan")
    out = reduce_last_dim(x, "sum")
    assert torch.isnan(out[0]), "NaN row should produce NaN sum."
    assert not torch.isnan(out[1]), "Clean row should not produce NaN."


def test_3d_input():
    """Batch dim (B, M, N) is supported via view in the wrapper."""
    x = torch.randn(4, 32, 64, dtype=torch.float32, device=DEVICE)
    torch.testing.assert_close(
        reduce_last_dim(x, "mean"),
        x.mean(dim=-1),
        atol=1e-4,
        rtol=1e-4,
    )


# ---------------------------------------------------------------------------
# Benchmark (required by self_test_tasks.md for task 2).
# ---------------------------------------------------------------------------
def test_benchmark():
    """Compare NineToothed mean-reduction vs PyTorch on a realistic shape.

    Uses the skill's own `scripts/bench_compare.py` (do_bench path) so the
    self-test needs no external pytest-benchmark plugin.
    """
    _sys.path.insert(
        0, str(_pathlib.Path(__file__).resolve().parent.parent.parent / "scripts")
    )
    from bench_compare import benchmark as bench

    M, N = 1024, 2048
    x = torch.randn(M, N, dtype=torch.float16, device=DEVICE)

    nt = bench(lambda: reduce_last_dim(x, "mean"))
    pt = bench(lambda: x.mean(dim=-1))
    print(
        f"\nreduction mean ({M}x{N} fp16): nt {nt['mean_ms']:.4f} ms "
        f"({nt['timer']}), torch {pt['mean_ms']:.4f} ms"
    )

    # Correctness check after benchmark.
    torch.testing.assert_close(
        reduce_last_dim(x, "mean"),
        x.float().mean(dim=-1).half(),
        atol=1e-2,
        rtol=1e-2,
    )
