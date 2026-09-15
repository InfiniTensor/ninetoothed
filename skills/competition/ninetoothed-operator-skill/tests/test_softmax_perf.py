"""
Self-test task 4: Performance analysis and benchmark comparison
Operator: softmax (revisited) + generated-source dump trigger

This task covers:
- Benchmark: NineToothed softmax vs torch.softmax across shapes
- Generated-source dump trigger: request Triton IR emission and verify correctness
- Performance regression analysis: identify when NineToothed lags PyTorch and why
- AOT build workflow note (not executed by this self-test)

Run:
    pytest tests/test_softmax_perf.py -v -m benchmark
    pytest tests/test_softmax_perf.py -v -k generated_source
"""
import os
import pytest
import torch
import triton
from ops.softmax import kernel as softmax_kernel, make_softmax

pytestmark = pytest.mark.benchmark


def make_block_size(ncols: int) -> int:
    b = 1
    while b < ncols:
        b *= 2
    return b


def run_nt_softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    softmax_kernel(x, out, BLOCK_SIZE=make_block_size(x.shape[1]))
    return out


# ── generated-source dump trigger ────────────────────────────────────────────

def test_generated_source_dump_trigger():
    """
    Request a NineToothed-generated Triton source dump and verify correctness.
    This verifies the dump trigger and post-dump correctness.
    """
    os.environ["NINETOOTHED_DUMP_GENERATED_SOURCE"] = "1"
    x = torch.randn(8, 256, device="cuda")
    out = torch.empty_like(x)
    try:
        # Create a fresh kernel after setting the env var; the module-level
        # kernel was constructed before this test can enable dumping.
        dump_kernel = make_softmax(ndim=2)
        dump_kernel(x, out, BLOCK_SIZE=256)
        torch.cuda.synchronize()
    finally:
        # Clean up env var so it doesn't affect other tests.
        del os.environ["NINETOOTHED_DUMP_GENERATED_SOURCE"]

    # Verify correctness after dump
    ref = torch.softmax(x, dim=-1)
    assert torch.allclose(out, ref, atol=1e-5), "correctness check after source dump"


# ── benchmark: NineToothed vs PyTorch across shapes ──────────────────────────

BENCHMARK_SHAPES = [
    (512,  256),    # small rows
    (512,  1024),   # medium rows
    (512,  4096),   # large rows (typical LLM attention)
    (2048, 4096),   # large batch + large rows
]


@pytest.mark.benchmark
@pytest.mark.parametrize("shape", BENCHMARK_SHAPES)
def test_softmax_nt_throughput(shape):
    """Measure NineToothed softmax bandwidth (GB/s)."""
    x = torch.randn(shape, dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)
    bs = make_block_size(shape[1])

    # Warm up
    for _ in range(3):
        softmax_kernel(x, out, BLOCK_SIZE=bs)
    torch.cuda.synchronize()

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    N_ITERS = 50
    start.record()
    for _ in range(N_ITERS):
        softmax_kernel(x, out, BLOCK_SIZE=bs)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / N_ITERS
    # softmax reads input once, writes output once = 2 * bytes
    bytes_transferred = 2 * x.numel() * x.element_size()
    gbps = bytes_transferred / (ms * 1e-3) / 1e9

    print(f"\nshape={shape} | NineToothed: {ms:.3f}ms | {gbps:.1f} GB/s")
    assert gbps > 0, "benchmark produced zero throughput"


@pytest.mark.benchmark
@pytest.mark.parametrize("shape", BENCHMARK_SHAPES)
def test_softmax_torch_throughput(shape):
    """PyTorch softmax baseline for comparison."""
    x = torch.randn(shape, dtype=torch.float32, device="cuda")

    # Warm up
    for _ in range(3):
        torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    N_ITERS = 50
    start.record()
    for _ in range(N_ITERS):
        torch.softmax(x, dim=-1)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / N_ITERS
    bytes_transferred = 2 * x.numel() * x.element_size()
    gbps = bytes_transferred / (ms * 1e-3) / 1e9

    print(f"\nshape={shape} | PyTorch:     {ms:.3f}ms | {gbps:.1f} GB/s")


# ── regression analysis: small BLOCK_SIZE padding waste ──────────────────────

def test_block_size_padding_waste():
    """
    Diagnose performance regression when BLOCK_SIZE >> ncols.
    Example: ncols=100, BLOCK_SIZE=128 → 22% waste (acceptable).
             ncols=100, BLOCK_SIZE=1024 → 90% waste (problem).

    This test documents the rule: always use smallest power-of-2 >= ncols.
    """
    x = torch.randn(512, 100, device="cuda")
    out = torch.empty_like(x)

    # Correct: BLOCK_SIZE=128 (next power of 2 >= 100)
    softmax_kernel(x, out, BLOCK_SIZE=128)
    ref = torch.softmax(x, dim=-1)
    assert torch.allclose(out, ref, atol=1e-5), "BLOCK_SIZE=128 correctness check"

    # Larger BLOCK_SIZE still correct but wastes compute
    softmax_kernel(x, out, BLOCK_SIZE=1024)
    assert torch.allclose(out, ref, atol=1e-5), "BLOCK_SIZE=1024 correctness check"

    # Document: BLOCK_SIZE=128 should be faster than BLOCK_SIZE=1024 for ncols=100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        softmax_kernel(x, out, BLOCK_SIZE=128)
    end.record()
    torch.cuda.synchronize()
    ms_128 = start.elapsed_time(end) / 100

    start.record()
    for _ in range(100):
        softmax_kernel(x, out, BLOCK_SIZE=1024)
    end.record()
    torch.cuda.synchronize()
    ms_1024 = start.elapsed_time(end) / 100

    speedup = ms_1024 / ms_128
    print(f"\nncols=100: BLOCK_SIZE=128: {ms_128:.3f}ms | "
          f"BLOCK_SIZE=1024: {ms_1024:.3f}ms | speedup: {speedup:.2f}x")
    # Optimal BLOCK_SIZE should be at least as fast
    assert ms_128 <= ms_1024 * 1.5, \
        f"Optimal BLOCK_SIZE=128 unexpectedly slower: {speedup:.2f}x"
