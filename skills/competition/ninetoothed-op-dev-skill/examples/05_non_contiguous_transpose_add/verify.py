#!/usr/bin/env python3
"""Fork-runnable verify for example 05 (non-contiguous add)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_EX = Path(__file__).resolve().parent
sys.path.insert(0, str(_EX / "solution"))

from strided_add_kernel import add_strided  # noqa: E402


def check_transpose_add(device: str = "cuda") -> None:
    m, n = 64, 48
    base_a = torch.randn(m, n, device=device)
    base_b = torch.randn(m, n, device=device)
    lhs = base_a.t().contiguous().t()
    rhs = base_b.t().contiguous().t()
    assert not lhs.is_contiguous()
    assert not rhs.is_contiguous()
    out = add_strided(lhs, rhs)
    expected = lhs + rhs
    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)
    print(
        f"correctness: PASS shape={tuple(out.shape)} "
        f"lhs_stride={tuple(lhs.stride())} contiguous={lhs.is_contiguous()}"
    )


def check_empty_strided(device: str = "cuda") -> None:
    shape = (64, 48)
    strides = (96, 1)
    lhs = torch.empty_strided(shape, strides, device=device)
    rhs = torch.empty_strided(shape, strides, device=device)
    lhs.copy_(torch.randn(shape, device=device))
    rhs.copy_(torch.randn(shape, device=device))
    assert not lhs.is_contiguous()
    out = add_strided(lhs, rhs)
    assert torch.allclose(out, lhs + rhs, atol=1e-5, rtol=1e-5)
    print(f"correctness: PASS empty_strided strides={strides}")


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA required for this example kernel")
        return 0
    check_transpose_add()
    check_empty_strided()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
