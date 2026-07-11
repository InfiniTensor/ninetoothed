"""Minimal row-wise softmax kernel for example 03 (fork-runnable demo)."""

from __future__ import annotations

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((1, BLOCK_SIZE)), output.tile((1, BLOCK_SIZE))


def application(input, output):
    input_loaded = input
    row_minus_max = input_loaded - ntl.max(input_loaded)
    numerator = ntl.exp(row_minus_max)
    denominator = ntl.sum(numerator)
    output = numerator / denominator  # noqa: F841


_tensors = (Tensor(2, other=float("-inf")), Tensor(2))
_kernel = ninetoothed.make(arrangement, application, _tensors)


def softmax(input: torch.Tensor, *, block_size: int | None = None) -> torch.Tensor:
    if block_size is None:
        block_size = int(input.shape[-1])
    output = torch.empty_like(input)
    _kernel(input, output, BLOCK_SIZE=block_size)
    return output
