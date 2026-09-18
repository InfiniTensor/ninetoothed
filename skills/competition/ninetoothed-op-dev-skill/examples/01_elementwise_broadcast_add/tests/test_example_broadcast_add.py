"""Correctness test for example 01 broadcast add."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_EX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EX / "solution"))

from broadcast_add import broadcast_add  # noqa: E402


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("m,n", [(8, 16), (32, 64)])
def test_broadcast_add_matches_torch(m: int, n: int):
    device = _device()
    if device == "cpu":
        pytest.skip("NineToothed kernels typically require CUDA; skip CPU")
    a = torch.randn(m, 1, dtype=torch.float32, device=device)
    b = torch.randn(1, n, dtype=torch.float32, device=device)
    out = broadcast_add(a, b)
    ref = torch.add(a, b)
    assert out.shape == (m, n)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_broadcast_add_reuses_out_buffer():
    device = _device()
    if device == "cpu":
        pytest.skip("NineToothed kernels typically require CUDA; skip CPU")
    m, n = 16, 32
    a = torch.randn(m, 1, dtype=torch.float32, device=device)
    b = torch.randn(1, n, dtype=torch.float32, device=device)
    out_buffer = torch.empty((m, n), device=device, dtype=a.dtype)
    returned = broadcast_add(a, b, out=out_buffer)
    assert returned is out_buffer
    assert torch.allclose(out_buffer, torch.add(a, b), atol=1e-5, rtol=1e-5)


def test_broadcast_add_rejects_bad_out_shape():
    device = _device()
    if device == "cpu":
        pytest.skip("NineToothed kernels typically require CUDA; skip CPU")
    m, n = 16, 32
    a = torch.randn(m, 1, dtype=torch.float32, device=device)
    b = torch.randn(1, n, dtype=torch.float32, device=device)
    bad = torch.empty((m, n + 1), device=device, dtype=a.dtype)
    with pytest.raises(ValueError, match="out shape"):
        broadcast_add(a, b, out=bad)


def test_broadcast_add_float16_matches_torch():
    """Minimal float16 correctness; benchmark remains float32-only."""
    device = _device()
    if device == "cpu":
        pytest.skip("NineToothed kernels typically require CUDA; skip CPU")
    m, n = 8, 16
    a = torch.randn(m, 1, dtype=torch.float16, device=device)
    b = torch.randn(1, n, dtype=torch.float16, device=device)
    out = broadcast_add(a, b)
    ref = torch.add(a, b)
    assert out.dtype == torch.float16
    assert out.shape == (m, n)
    assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
