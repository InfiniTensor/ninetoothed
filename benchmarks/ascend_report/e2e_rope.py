"""Generated NineToothed kernels for additional inference replacements."""


import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def rotary_embedding(x, cos, sin):
    """Run the pinned generated pair kernel and reassemble the full tensor."""
    import importlib

    if x.ndim != 4 or x.shape[-1] % 2 or cos.ndim != 4 or sin.ndim != 4:
        raise ValueError(
            "rotary_embedding expects rank-four x and broadcastable cos/sin"
        )

    module = importlib.import_module("ntops_lab.kernels.fused.general.rotary_embedding")
    half = x.shape[-1] // 2
    target_shape = x.shape[:-1] + (half,)
    cos = cos.expand(target_shape).contiguous().view(-1)
    sin = sin.expand(target_shape).contiguous().view(-1)
    x0 = x[..., :half].contiguous().view(-1)
    x1 = x[..., half:].contiguous().view(-1)
    out0, out1 = module.run(x0, x1, cos, sin)

    return torch.cat((out0.view(target_shape), out1.view(target_shape)), dim=-1)


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
