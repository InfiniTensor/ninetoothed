"""Generated grouped-query attention head expansion."""

import functools

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(out, source, repeats):
    return out.tile((1, 1, 1, -1)), source.tile((1, 1, 1, -1)), repeats


def application(out, source, repeats):
    source_head = out.offsets(-3)[None, :, None, None] // repeats
    pointer = (
        source.source.data_ptr()
        + out.offsets(-4)[:, None, None, None] * source.source.stride(0)
        + source_head * source.source.stride(1)
        + out.offsets(-2)[None, None, :, None] * source.source.stride(2)
        + out.offsets(-1)[None, None, None, :] * source.source.stride(3)
    )
    out = ntl.load(pointer)  # noqa: F841


@functools.cache
def kernel(repeats):
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(4), Tensor(4), Tensor(0, constexpr=True)),
        kernel_name=f"e2e_repeat_kv_heads_r{repeats}",
        max_num_configs=1,
    )


def repeat_kv_heads(x, repeats):
    if x.ndim != 4 or repeats == 1:
        raise ValueError("repeat_kv_heads expects rank-four input and repeats > 1")
    out = torch.empty(
        (x.shape[0], x.shape[1] * repeats, x.shape[2], x.shape[3]),
        device=x.device,
        dtype=x.dtype,
    )
    kernel(repeats)(out, x, repeats)
    return out
