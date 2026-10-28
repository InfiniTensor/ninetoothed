"""Generated layout kernels used by the stage-seven attention output."""

import functools

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def _attention_arrangement(out, source, head_dim):
    return out.tile((1, 1, -1)), source.tile((1, 1, 1, -1)), head_dim


def _attention_application(out, source, head_dim):
    batch = out.offsets(-3)[:, None, None]
    token = out.offsets(-2)[None, :, None]
    dimension = out.offsets(-1)[None, None, :]
    head = dimension // head_dim
    head_dimension = dimension % head_dim
    offset = (
        batch * source.source.stride(0)
        + head * source.source.stride(1)
        + token * source.source.stride(2)
        + head_dimension * source.source.stride(3)
    )
    out = ntl.load(source.source.data_ptr() + offset)  # noqa: F841


@functools.cache
def _attention_kernel(dtype_name):
    return ninetoothed.make(
        _attention_arrangement,
        _attention_application,
        (
            Tensor(3, dtype=getattr(torch, dtype_name)),
            Tensor(4, dtype=getattr(torch, dtype_name)),
            Tensor(0, constexpr=True),
        ),
        kernel_name=f"e2e_attention_output_{dtype_name}",
        max_num_configs=1,
    )


def attention_output(source):
    """Merge [batch, heads, tokens, width] into [batch, tokens, heads*width]."""
    if source.ndim != 4:
        raise ValueError("attention_output expects rank-four attention output")

    shape = (source.shape[0], source.shape[2], source.shape[1] * source.shape[3])
    out = torch.empty(shape, device=source.device, dtype=source.dtype)
    _attention_kernel(str(source.dtype).removeprefix("torch."))(
        out, source, source.shape[-1]
    )
    return out


def _flat_arrangement(out, source):
    return out.tile((1024,)), source.tile((1024,))


def _flat_application(out, source):
    out = source  # noqa: F841


@functools.cache
def _flat_kernel(dtype_name):
    return ninetoothed.make(
        _flat_arrangement,
        _flat_application,
        (
            Tensor(1, dtype=getattr(torch, dtype_name)),
            Tensor(1, dtype=getattr(torch, dtype_name)),
        ),
        kernel_name=f"e2e_copy_flat_{dtype_name}",
        max_num_configs=1,
    )


def copy_flat(source, destination):
    if source.shape != destination.shape or source.dtype != destination.dtype:
        raise ValueError("Generated flat copy requires matching shape and dtype")
    _flat_kernel(str(source.dtype).removeprefix("torch."))(destination, source)


def _iota_arrangement(out):
    return (out.tile((1024,)),)


def _iota_application(out):
    out = out.offsets(-1).to(ntl.int64)  # noqa: F841


_iota_kernel = ninetoothed.make(
    _iota_arrangement,
    _iota_application,
    (Tensor(1, dtype=torch.int64),),
    kernel_name="e2e_iota_int64",
    max_num_configs=1,
)


def iota(length):
    out = torch.empty((length,), dtype=torch.int64, device="npu")
    _iota_kernel(out)
    return out
