#!/usr/bin/env python3
"""Fork-runnable verify for example 03 (correctness vs torch.softmax)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_EX = Path(__file__).resolve().parent
sys.path.insert(0, str(_EX / "solution"))

from softmax_kernel import softmax  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA required for this example kernel")
        return 0

    device = "cuda"
    m, n = 64, 128
    x = torch.rand((m, n), dtype=torch.float32, device=device)
    out = softmax(x)
    expected = torch.softmax(x, dim=-1)
    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)
    max_err = (out - expected).abs().max().item()
    print(f"correctness: PASS shape={tuple(out.shape)} max_err={max_err:.2e}")
    print(
        "note: prefer upstream `pytest tests/test_softmax.py` when available in the fork"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
