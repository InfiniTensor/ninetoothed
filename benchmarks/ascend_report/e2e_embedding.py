"""Generated embedding row gather used by the experimental full-replacement stage."""

import functools

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(out, index, weight):
    return out.tile((1, -1)), index.tile((1,)), weight.tile((1, -1))


def application(out, index, weight):
    token = ntl.load(index.source.data_ptr()).to(ntl.int64)
    pointer = (
        weight.source.data_ptr()
        + token * weight.source.stride(0)
        + out.offsets(-1)[None, :] * weight.source.stride(1)
    )
    out = ntl.load(pointer)  # noqa: F841


@functools.cache
def _kernel():
    return ninetoothed.make(
        arrangement,
        application,
        (
            Tensor(2),
            Tensor(1, dtype=torch.int64),
            Tensor(2),
        ),
        kernel_name="e2e_embedding_row_gather",
        max_num_configs=1,
    )


def embedding(weight, indices):
    if weight.ndim != 2 or indices.ndim < 1:
        raise ValueError(
            "Generated embedding expects a rank-two weight and indexed indices"
        )

    output = torch.empty(
        (*indices.shape, weight.shape[1]),
        device=weight.device,
        dtype=weight.dtype,
    )
    flat_output = output.view(-1, weight.shape[1])
    flat_indices = indices.reshape(-1)
    kernel = _kernel()
    for row in range(flat_indices.shape[0]):
        kernel(
            flat_output[row : row + 1],
            flat_indices[row : row + 1],
            weight,
        )
    return output
