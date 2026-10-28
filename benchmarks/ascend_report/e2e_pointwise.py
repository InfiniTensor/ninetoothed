"""Direct generated binary kernels for same-shape model tensors."""

import functools

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def _arrangement(out, lhs, rhs):
    return out.tile((1, 1, -1)), lhs.tile((1, 1, -1)), rhs.tile((1, 1, -1))


def _add_application(out, lhs, rhs):
    lhs_offset = (
        out.offsets(-3)[:, None, None] * lhs.source.stride(0)
        + out.offsets(-2)[None, :, None] * lhs.source.stride(1)
        + out.offsets(-1)[None, None, :] * lhs.source.stride(2)
    )
    rhs_offset = (
        out.offsets(-3)[:, None, None] * rhs.source.stride(0)
        + out.offsets(-2)[None, :, None] * rhs.source.stride(1)
        + out.offsets(-1)[None, None, :] * rhs.source.stride(2)
    )
    lhs_value = ntl.load(lhs.source.data_ptr() + lhs_offset)
    rhs_value = ntl.load(rhs.source.data_ptr() + rhs_offset)
    out = lhs_value + rhs_value  # noqa: F841


def _mul_application(out, lhs, rhs):
    lhs_offset = (
        out.offsets(-3)[:, None, None] * lhs.source.stride(0)
        + out.offsets(-2)[None, :, None] * lhs.source.stride(1)
        + out.offsets(-1)[None, None, :] * lhs.source.stride(2)
    )
    rhs_offset = (
        out.offsets(-3)[:, None, None] * rhs.source.stride(0)
        + out.offsets(-2)[None, :, None] * rhs.source.stride(1)
        + out.offsets(-1)[None, None, :] * rhs.source.stride(2)
    )
    lhs_value = ntl.load(lhs.source.data_ptr() + lhs_offset)
    rhs_value = ntl.load(rhs.source.data_ptr() + rhs_offset)
    out = lhs_value * rhs_value  # noqa: F841


_APPLICATIONS = {"add": _add_application, "mul": _mul_application}


@functools.cache
def _kernel(operation, dtype_name):
    return ninetoothed.make(
        _arrangement,
        _APPLICATIONS[operation],
        (
            Tensor(3, dtype=getattr(torch, dtype_name)),
            Tensor(3, dtype=getattr(torch, dtype_name)),
            Tensor(3, dtype=getattr(torch, dtype_name)),
        ),
        kernel_name=f"e2e_{operation}_direct_{dtype_name}",
        max_num_configs=1,
    )


def binary(operation, lhs, rhs):
    if operation not in _APPLICATIONS:
        raise ValueError("Generated binary supports only add and mul.")
    if (
        lhs.ndim != 3
        or rhs.ndim != 3
        or lhs.shape != rhs.shape
        or lhs.dtype != rhs.dtype
        or lhs.device != rhs.device
    ):
        raise ValueError("Generated binary requires two same-shape rank-three tensors.")

    out = torch.empty_like(lhs)
    _kernel(operation, str(lhs.dtype).removeprefix("torch."))(out, lhs, rhs)
    return out
