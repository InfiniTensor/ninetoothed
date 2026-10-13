"""
Self-test task 3: Non-contiguous input handling
Operator: rms_norm
Focus: verify correct results for contiguous, transposed, row-sliced,
       and col-sliced tensors; document copy overhead for non-contiguous.

Run:
    pytest tests/test_rms_norm_noncontiguous.py -v -s
"""
import pytest
import torch
from ops.rms_norm import rms_norm


def torch_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_fp32 = x.float()
    rms = torch.sqrt((x_fp32 ** 2).mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 / rms).to(x.dtype)


# ── contiguous baseline ───────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", [
    (32, 128),
    (512, 512),
    (1024, 768),
])
def test_rms_norm_contiguous(shape):
    x = torch.randn(shape, dtype=torch.float32, device="cuda")
    assert x.is_contiguous()
    out = rms_norm(x)
    ref = torch_rms_norm(x)
    assert torch.allclose(out.float(), ref.float(), atol=1e-4), \
        f"contiguous mismatch: max_err={(out.float()-ref.float()).abs().max().item():.6f}"


# ── non-contiguous: transposed ────────────────────────────────────────────────

def test_rms_norm_transposed():
    x = torch.randn(64, 128, device="cuda").t()
    assert not x.is_contiguous()
    out = rms_norm(x)
    ref = torch_rms_norm(x.contiguous())
    assert torch.allclose(out.float(), ref.float(), atol=1e-4)


# ── non-contiguous: row slice ─────────────────────────────────────────────────

def test_rms_norm_row_slice():
    x = torch.randn(256, 128, device="cuda")[::2]
    assert not x.is_contiguous()
    out = rms_norm(x)
    ref = torch_rms_norm(x.contiguous())
    assert torch.allclose(out.float(), ref.float(), atol=1e-4)


# ── non-contiguous: col slice ─────────────────────────────────────────────────

def test_rms_norm_col_slice():
    x = torch.randn(128, 512, device="cuda")[:, ::2]
    assert not x.is_contiguous()
    out = rms_norm(x)
    ref = torch_rms_norm(x.contiguous())
    assert torch.allclose(out.float(), ref.float(), atol=1e-4)


def test_rms_norm_rejects_dynamic_eps():
    x = torch.randn(16, 64, device="cuda")
    with pytest.raises(ValueError, match="Only eps=1e-6"):
        rms_norm(x, eps=1e-5)


# ── contiguous vs non-contiguous overhead (manual timing) ────────────────────

@pytest.mark.benchmark
def test_rms_norm_copy_overhead():
    """
    Measure overhead of .contiguous() copy for non-contiguous inputs.
    Documents the known trade-off: non-contiguous path = copy + kernel.
    """
    shape = (1024, 768)
    x_cont = torch.randn(shape, device="cuda")
    x_noncont = torch.randn(shape[1], shape[0], device="cuda").t()
    assert not x_noncont.is_contiguous()

    N = 100

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(5): rms_norm(x_cont)
    torch.cuda.synchronize()
    start.record()
    for _ in range(N): rms_norm(x_cont)
    end.record()
    torch.cuda.synchronize()
    ms_cont = start.elapsed_time(end) / N

    for _ in range(5): rms_norm(x_noncont)
    torch.cuda.synchronize()
    start.record()
    for _ in range(N): rms_norm(x_noncont)
    end.record()
    torch.cuda.synchronize()
    ms_nc = start.elapsed_time(end) / N

    overhead = (ms_nc - ms_cont) / ms_cont * 100
    print(f"\ncontiguous: {ms_cont:.3f}ms | non-contiguous: {ms_nc:.3f}ms | overhead: {overhead:.1f}%")
    assert ms_nc < ms_cont * 4, f"overhead too high: {overhead:.1f}%"
