"""
Correctness and benchmark tests for softmax operator.

Run:
    pytest tests/test_softmax.py -v
    pytest tests/test_softmax.py -v -k benchmark -s
"""
import pytest
import torch
from ops.softmax import make_softmax


def run_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    k = make_softmax(ndim=2)
    output = torch.empty_like(x)
    k(x, output)    # no BLOCK_SIZE — autotuning handles it
    return output


# ── correctness ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", [
    (1, 128),
    (64, 256),
    (1823, 781),
    (512, 1),
    (4, 4096),
])
def test_softmax_correctness(shape):
    x = torch.rand(shape, dtype=torch.float32, device="cuda")
    out = run_softmax(x)
    ref = torch.softmax(x, dim=-1)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), \
        f"softmax mismatch shape={shape}: max_err={(out-ref).abs().max().item():.6f}"


def test_softmax_row_sums_to_one():
    x = torch.randn(128, 512, device="cuda")
    out = run_softmax(x)
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_softmax_numerical_stability_large_logits():
    x = torch.full((8, 256), 1e4, device="cuda")
    x[:, 0] = 1e6
    out = run_softmax(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_softmax_numerical_stability_negative_logits():
    x = -torch.abs(torch.randn(32, 128, device="cuda")) * 100
    out = run_softmax(x)
    assert not torch.isnan(out).any()
    assert torch.allclose(out.sum(dim=-1), torch.ones(32, device="cuda"), atol=1e-3)


def test_softmax_noncontiguous_rows_fallback():
    x_noncontig = torch.randn(128, 256, device="cuda")[::2]
    assert not x_noncontig.is_contiguous()
    x = x_noncontig.contiguous()
    out = run_softmax(x)
    ref = torch.softmax(x_noncontig, dim=-1)
    assert torch.allclose(out, ref, atol=1e-5)


# ── benchmark (manual CUDA timing) ───────────────────────────────────────────

@pytest.mark.parametrize("shape", [
    (1024, 512),
    (1024, 2048),
    (4096, 4096),
])
@pytest.mark.benchmark
def test_softmax_benchmark(shape):
    k = make_softmax(ndim=2)
    x = torch.randn(shape, dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)

    for _ in range(3):
        k(x, out)
    torch.cuda.synchronize()

    N = 50
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(N):
        k(x, out)
    end.record()
    torch.cuda.synchronize()
    ms_nt = start.elapsed_time(end) / N

    start.record()
    for _ in range(N):
        torch.softmax(x, dim=-1)
    end.record()
    torch.cuda.synchronize()
    ms_torch = start.elapsed_time(end) / N

    gbps_nt = 2 * x.nelement() * x.element_size() / ms_nt * 1e-6
    gbps_torch = 2 * x.nelement() * x.element_size() / ms_torch * 1e-6
    print(f"\nshape={shape} | NineToothed: {ms_nt:.3f}ms {gbps_nt:.1f}GB/s "
          f"| PyTorch: {ms_torch:.3f}ms {gbps_torch:.1f}GB/s")
    assert ms_nt > 0
