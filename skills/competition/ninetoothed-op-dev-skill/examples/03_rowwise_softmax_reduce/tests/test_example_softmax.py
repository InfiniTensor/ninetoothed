"""Correctness test for example 03 row-wise softmax (optional local demo)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_EX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EX / "solution"))

from softmax_kernel import softmax  # noqa: E402


def test_softmax_matches_torch():
    if not torch.cuda.is_available():
        pytest.skip("NineToothed kernels typically require CUDA; skip CPU")
    m, n = 32, 64
    x = torch.rand((m, n), dtype=torch.float32, device="cuda")
    out = softmax(x)
    expected = torch.softmax(x, dim=-1)
    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)
