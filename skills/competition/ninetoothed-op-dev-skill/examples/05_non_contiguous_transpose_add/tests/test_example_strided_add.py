"""Correctness test for example 05 non-contiguous add."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_EX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EX / "solution"))

from strided_add_kernel import add_strided  # noqa: E402


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_transpose_view_add():
    device = _device()
    if device == "cpu":
        pytest.skip("NineToothed kernels typically require CUDA; skip CPU")
    m, n = 64, 48
    base_a = torch.randn(m, n, device=device)
    base_b = torch.randn(m, n, device=device)
    lhs = base_a.t().contiguous().t()
    rhs = base_b.t().contiguous().t()
    assert not lhs.is_contiguous()
    assert not rhs.is_contiguous()
    out = add_strided(lhs, rhs)
    ref = lhs + rhs
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_empty_strided_add():
    device = _device()
    if device == "cpu":
        pytest.skip("NineToothed kernels typically require CUDA; skip CPU")
    shape = (64, 48)
    strides = (96, 1)
    lhs = torch.empty_strided(shape, strides, device=device)
    rhs = torch.empty_strided(shape, strides, device=device)
    lhs.copy_(torch.randn(shape, device=device))
    rhs.copy_(torch.randn(shape, device=device))
    assert not lhs.is_contiguous()
    out = add_strided(lhs, rhs)
    assert torch.allclose(out, lhs + rhs, atol=1e-5, rtol=1e-5)
