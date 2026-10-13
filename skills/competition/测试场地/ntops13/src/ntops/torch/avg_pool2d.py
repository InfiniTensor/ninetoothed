import torch

import ntops
from ntops.torch.pooling import _calculate_output_size
from ntops.torch.utils import _cached_make


def avg_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    if stride is None:
        stride = kernel_size

    if isinstance(stride, int):
        stride = (stride, stride)

    if isinstance(padding, int):
        padding = (padding, padding)

    assert not ceil_mode, "`ceil_mode` is not supported yet."

    assert count_include_pad, "`count_include_pad` is not supported yet."

    assert divisor_override is None, "`divisor_override` is not supported yet."

    n, c, h, w = input.shape

    h_ = _calculate_output_size(
        h, kernel_size[0], stride=stride[0], padding=padding[0], ceil_mode=ceil_mode
    )
    w_ = _calculate_output_size(
        w, kernel_size[1], stride=stride[1], padding=padding[1], ceil_mode=ceil_mode
    )

    output = torch.empty((n, c, h_, w_), dtype=input.dtype, device=input.device)

    kernel = _cached_make(
        ntops.kernels.avg_pool2d.premake,
        dilation_h=1,
        dilation_w=1,
        ceil_mode=ceil_mode,
    )

    kernel(
        input,
        output,
        kernel_size_h=kernel_size[0],
        kernel_size_w=kernel_size[1],
        stride_h=stride[0],
        stride_w=stride[1],
        padding_h=padding[0],
        padding_w=padding[1],
    )

    return output
