"""
Correctness tests for relu operator.

Run:
    pytest tests/test_relu.py -v
    pytest tests/test_relu.py -v -k noncontiguous
"""
import pytest
import torch
from ops.relu import make_relu


def run_relu(x: torch.Tensor) -> torch.Tensor:
    """Run relu via the documented contiguous-copy fallback path."""
    k = make_relu(ndim=1)
    x_flat = x.contiguous().flatten()
    out_flat = torch.empty_like(x_flat)
    k(x_flat, out_flat)           # no BLOCK_SIZE — autotuning handles it
    return out_flat.reshape(x.shape)


# ── correctness: shape sweep ────────────────────────────────────────────────

@pytest.mark.parametrize("size", [63, 1024, 3333, 65536])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_relu_correctness(size, dtype):
    x = torch.randn(size, dtype=dtype, device="cuda")
    out = run_relu(x)
    ref = torch.relu(x)
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    assert torch.allclose(out, ref, atol=atol, rtol=1e-3), \
        f"relu mismatch: max_err={(out - ref).abs().max().item():.6f}"


def test_relu_all_negative():
    x = -torch.abs(torch.randn(512, device="cuda"))
    out = run_relu(x)
    assert (out == 0).all()


def test_relu_all_positive():
    x = torch.abs(torch.randn(512, device="cuda"))
    out = run_relu(x)
    assert torch.allclose(out, x, atol=1e-6)


# ── correctness: non-contiguous fallback path ────────────────────────────────

def test_relu_noncontiguous_stride2_fallback():
    base = torch.randn(2048, device="cuda")
    x = base[::2]
    assert not x.is_contiguous()
    x_cont = x.contiguous()
    k = make_relu(ndim=1)
    out = torch.empty_like(x_cont)
    k(x_cont, out)
    ref = torch.relu(x_cont)
    assert torch.allclose(out, ref, atol=1e-5)


def test_relu_noncontiguous_transposed_fallback():
    x = torch.randn(64, 32, device="cuda").t()
    assert not x.is_contiguous()
    out = run_relu(x)
    ref = torch.relu(x)
    assert torch.allclose(out, ref, atol=1e-5)


# ── benchmark (manual timing, no pytest-benchmark plugin needed) ──────────────

@pytest.mark.parametrize("size", [2**20, 2**24])
@pytest.mark.benchmark
def test_relu_benchmark(size):
    k = make_relu(ndim=1)
    x = torch.randn(size, dtype=torch.float16, device="cuda")
    out = torch.empty_like(x)

    # Warm up
    for _ in range(5):
        k(x, out)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    N = 50
    start.record()
    for _ in range(N):
        k(x, out)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / N
    gbps = 2 * x.nelement() * x.element_size() / ms * 1e-6
    print(f"\nrelu size={size}: {ms:.3f}ms | {gbps:.1f} GB/s")
    assert ms > 0
