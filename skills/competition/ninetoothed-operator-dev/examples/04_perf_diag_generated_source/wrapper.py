"""Wrappers for both the bad and fixed softmax kernels."""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

import torch

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import kernel_bad as _bad_mod
import kernel_fixed as _fixed_mod

_bad = _bad_mod.kernel_bad
_fixed = _fixed_mod.kernel_fixed


def softmax_bad(x: torch.Tensor) -> torch.Tensor:
    """Numerically unstable softmax — for demo / regression diagnosis."""
    x = x.contiguous()
    out = torch.empty_like(x)
    _bad(x, out, BLOCK_SIZE=x.shape[-1])

    return out


def softmax_fixed(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax with fp32 accumulation."""
    x = x.contiguous()
    out = torch.empty_like(x)
    _fixed(x, out, BLOCK_SIZE=x.shape[-1])

    return out
