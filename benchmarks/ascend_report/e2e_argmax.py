"""Generated greedy argmax used by the additional inference stage."""

import functools

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

CHUNK_SIZE = 2048


def _chunk_arrangement(x, maxima, indices):
    return x.tile((1, CHUNK_SIZE)), maxima.tile((1,)), indices.tile((1,))


def _chunk_application(x, maxima, indices):
    maximum = ntl.max(x, axis=1)
    valid = (x.offsets(-1) < x.source.shape[-1]) & (x == maximum[:, None])
    candidates = ntl.where(
        valid,
        x.offsets(-1).to(ntl.int64),
        x.source.shape[-1],
    )
    maxima = maximum  # noqa: F841
    indices = ntl.min(candidates, axis=1)  # noqa: F841


@functools.cache
def _chunk_kernel(dtype_name):
    return ninetoothed.make(
        _chunk_arrangement,
        _chunk_application,
        (
            Tensor(2, dtype=getattr(torch, dtype_name), other=float("-inf")),
            Tensor(1, dtype=torch.float32),
            Tensor(1, dtype=torch.int64),
        ),
        kernel_name=f"e2e_argmax_chunk_{dtype_name}",
        max_num_configs=1,
    )


def _final_arrangement(maxima, indices, out):
    return maxima.tile((1, -1)), indices.tile((1, -1)), out.tile((1,))


def _final_application(maxima, indices, out):
    maximum = ntl.max(maxima, axis=1)
    penalty = (maxima < maximum).to(ntl.int64)
    candidates = penalty * 2147483647 + indices
    out = ntl.min(candidates, axis=1)  # noqa: F841


@functools.cache
def _final_kernel():
    return ninetoothed.make(
        _final_arrangement,
        _final_application,
        (
            Tensor(2, dtype=torch.float32, other=float("-inf")),
            Tensor(2, dtype=torch.int64),
            Tensor(1, dtype=torch.int64),
        ),
        kernel_name="e2e_argmax_final",
        max_num_configs=1,
    )


def argmax(scores):
    if scores.ndim != 1:
        raise ValueError("The generated argmax currently expects one flattened row")
    if scores.shape[-1] == 0:
        raise ValueError("Cannot take argmax of an empty tensor")

    chunk_count = (scores.shape[-1] + CHUNK_SIZE - 1) // CHUNK_SIZE
    chunk_maxima = torch.empty(
        (chunk_count,), device=scores.device, dtype=torch.float32
    )
    chunk_indices = torch.empty((chunk_count,), device=scores.device, dtype=torch.int64)
    _chunk_kernel(str(scores.dtype).removeprefix("torch."))(
        scores.view(1, -1), chunk_maxima, chunk_indices
    )

    output = torch.empty((1,), device=scores.device, dtype=torch.int64)
    _final_kernel()(chunk_maxima.view(1, -1), chunk_indices.view(1, -1), output)
    return output
