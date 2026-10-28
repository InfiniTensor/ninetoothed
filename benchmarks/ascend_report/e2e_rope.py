"""Generated NineToothed kernels for additional inference replacements."""

import functools

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def _direct_arrangement(out, source, cos, sin, half):
    return (
        out.tile((1, 1, 1, -1)),
        source.tile((1, 1, 1, -1)),
        cos.tile((1, 1, 1, -1)),
        sin.tile((1, 1, 1, -1)),
        half,
    )


def _direct_application(out, source, cos, sin, half):
    batch = out.offsets(-4)[:, None, None, None]
    head = out.offsets(-3)[None, :, None, None]
    token = out.offsets(-2)[None, None, :, None]
    dimension = out.offsets(-1)[None, None, None, :]
    second_half = dimension >= half

    source_offset = (
        batch * source.source.stride(0)
        + head * source.source.stride(1)
        + token * source.source.stride(2)
        + dimension * source.source.stride(3)
    )
    mate_offset = source_offset + ntl.where(
        second_half, -half, half
    ) * source.source.stride(3)
    frequency_dimension = dimension - ntl.where(second_half, half, 0)
    frequency_offset = (
        batch * cos.source.stride(0)
        + token * cos.source.stride(2)
        + frequency_dimension * cos.source.stride(3)
    )

    current = ntl.load(source.source.data_ptr() + source_offset)
    mate = ntl.load(source.source.data_ptr() + mate_offset)
    cos_value = ntl.load(cos.source.data_ptr() + frequency_offset)
    sin_value = ntl.load(sin.source.data_ptr() + frequency_offset)
    out = ntl.where(
        second_half,
        current * cos_value + mate * sin_value,
        current * cos_value - mate * sin_value,
    )  # noqa: F841


@functools.cache
def _direct_kernel(dtype_name):
    return ninetoothed.make(
        _direct_arrangement,
        _direct_application,
        (
            Tensor(4),
            Tensor(4, dtype=getattr(torch, dtype_name)),
            Tensor(4, dtype=torch.float32),
            Tensor(4, dtype=torch.float32),
            Tensor(0, constexpr=True),
        ),
        kernel_name=f"e2e_rotary_full_{dtype_name}",
        max_num_configs=1,
    )


def rotary_embedding(x, cos, sin):
    """Run one generated launch over the full rank-four Q/K tensor."""
    if x.ndim != 4 or x.shape[-1] % 2 or cos.ndim != 4 or sin.ndim != 4:
        raise ValueError(
            "rotary_embedding expects rank-four x and broadcastable cos/sin"
        )

    out = torch.empty_like(x)
    _direct_kernel(str(x.dtype).removeprefix("torch."))(
        out, x, cos, sin, x.shape[-1] // 2
    )
    return out


def _outer_arrangement(left, right, out):
    out_arranged = out.tile((64, 64))
    left_arranged = left.tile((64, 1)).tile((1, -1)).expand((-1, out_arranged.shape[1]))
    left_arranged.dtype = left_arranged.dtype.squeeze(0)
    right_arranged = (
        right.tile((1, 64)).tile((-1, 1)).expand((out_arranged.shape[0], -1))
    )
    right_arranged.dtype = right_arranged.dtype.squeeze(1)
    return left_arranged, right_arranged, out_arranged


def _outer_application(left, right, out):
    accumulator = ntl.zeros(out.shape, dtype=ntl.float32)
    for index in range(left.shape[0]):
        accumulator += ntl.dot(left[index], right[index])
    out = accumulator  # noqa: F841


_outer_kernel = ninetoothed.make(
    _outer_arrangement,
    _outer_application,
    (Tensor(2), Tensor(2), Tensor(2)),
    kernel_name="e2e_rope_frequency_outer",
    max_num_configs=1,
)


def frequency_outer(left, right):
    out = torch.empty(
        (left.shape[0], right.shape[0]), device=left.device, dtype=torch.float32
    )
    _outer_kernel(left[:, None], right[None, :], out)
    return out
