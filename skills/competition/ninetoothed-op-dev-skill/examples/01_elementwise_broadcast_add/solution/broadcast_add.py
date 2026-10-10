"""Minimal broadcast-add kernel for example 01 (fork-runnable demo)."""

from __future__ import annotations

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_M = Symbol("BLOCK_M", constexpr=True)
BLOCK_N = Symbol("BLOCK_N", constexpr=True)


def arrangement(a, b, out, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N):
    # Expand singleton dims to match out, then tile in lockstep (repo pattern).
    a = a.expand((-1, out.shape[1])).tile((BLOCK_M, BLOCK_N))
    b = b.expand((out.shape[0], -1)).tile((BLOCK_M, BLOCK_N))
    out = out.tile((BLOCK_M, BLOCK_N))
    return a, b, out


def application(a, b, out):
    out = ntl.cast(a, ntl.float32) + ntl.cast(b, ntl.float32)  # noqa: F841


_kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2), Tensor(2)))


def broadcast_add(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int = 32,
    block_n: int = 32,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    m, _ = a.shape
    _, n = b.shape

    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)

    if out.shape != (m, n):
        raise ValueError(f"out shape must be {(m, n)}, got {tuple(out.shape)}")
    if out.dtype != a.dtype:
        raise ValueError(f"out dtype must be {a.dtype}, got {out.dtype}")
    if out.device != a.device:
        raise ValueError(f"out device must be {a.device}, got {out.device}")

    _kernel(a, b, out, BLOCK_M=block_m, BLOCK_N=block_n)
    return out
