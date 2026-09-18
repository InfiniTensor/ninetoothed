"""Correctness test for example 09 block-size configs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_EX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EX / "solution"))

from add_tunable import add  # noqa: E402


@pytest.mark.parametrize("block_size", [32, 1024])
def test_add_block_size_matches_torch(block_size: int):
    if not torch.cuda.is_available():
        pytest.skip("NineToothed kernels typically require CUDA; skip CPU")
    n = 4096
    lhs = torch.rand(n, dtype=torch.float32, device="cuda")
    rhs = torch.rand(n, dtype=torch.float32, device="cuda")
    out = add(lhs, rhs, block_size=block_size)
    assert torch.allclose(out, lhs + rhs, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("block_size", [32, 256, 1024])
def test_add_reuses_out_buffer(block_size: int):
    if not torch.cuda.is_available():
        pytest.skip("NineToothed kernels typically require CUDA; skip CPU")
    n = 4096
    lhs = torch.rand(n, dtype=torch.float32, device="cuda")
    rhs = torch.rand(n, dtype=torch.float32, device="cuda")
    out_buffer = torch.empty_like(lhs)
    returned = add(lhs, rhs, block_size=block_size, out=out_buffer)
    assert returned is out_buffer
    assert torch.allclose(out_buffer, lhs + rhs, atol=1e-5, rtol=1e-5)


def test_add_rejects_bad_out_shape():
    if not torch.cuda.is_available():
        pytest.skip("NineToothed kernels typically require CUDA; skip CPU")
    n = 4096
    lhs = torch.rand(n, dtype=torch.float32, device="cuda")
    rhs = torch.rand(n, dtype=torch.float32, device="cuda")
    bad = torch.empty((n + 1,), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="out shape"):
        add(lhs, rhs, block_size=256, out=bad)
