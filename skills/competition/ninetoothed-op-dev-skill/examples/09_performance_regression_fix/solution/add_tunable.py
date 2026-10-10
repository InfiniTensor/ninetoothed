"""1-D add with tunable BLOCK_SIZE for example 09 (fork-runnable demo)."""

from __future__ import annotations

import torch

import ninetoothed
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(lhs, rhs, output, BLOCK_SIZE=BLOCK_SIZE):
    return (
        lhs.tile((BLOCK_SIZE,)),
        rhs.tile((BLOCK_SIZE,)),
        output.tile((BLOCK_SIZE,)),
    )


def application(lhs, rhs, output):
    output = lhs + rhs  # noqa: F841


_kernel = ninetoothed.make(arrangement, application, tuple(Tensor(1) for _ in range(3)))


def add(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    block_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(lhs)

    if out.shape != lhs.shape:
        raise ValueError(
            f"out shape must be {tuple(lhs.shape)}, got {tuple(out.shape)}"
        )
    if out.dtype != lhs.dtype:
        raise ValueError(f"out dtype must be {lhs.dtype}, got {out.dtype}")
    if out.device != lhs.device:
        raise ValueError(f"out device must be {lhs.device}, got {out.device}")

    _kernel(lhs, rhs, out, BLOCK_SIZE=block_size)
    return out
