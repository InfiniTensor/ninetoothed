"""Minimal strided add kernel for example 05 (fork-runnable demo)."""

from __future__ import annotations

import torch

import ninetoothed
from ninetoothed import Tensor


def arrangement(lhs, rhs, output):
    block_shape = (ninetoothed.block_size(), ninetoothed.block_size())
    return (
        lhs.tile(block_shape),
        rhs.tile(block_shape),
        output.tile(block_shape),
    )


def application(lhs, rhs, output):
    output = lhs + rhs  # noqa: F841


_kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2), Tensor(2)))


def add_strided(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Add two possibly non-contiguous tensors. Does not call .contiguous()."""
    output = torch.empty_like(lhs)
    _kernel(lhs, rhs, output)
    return output
