import torch

import ntops
from ntops.torch.pooling import _calculate_output_size
from ntops.torch.utils import _cached_make, _get_matmul_input_precision


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if isinstance(stride, int):
        stride = (stride, stride)

    if isinstance(padding, str):
        if padding == "valid":
            padding = 0

    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    assert groups == 1, "`groups` is not supported yet."

    n, c, h, w = input.shape
    k, _, r, s = weight.shape

    p = _calculate_output_size(
        h, r, stride=stride[0], padding=padding[0], dilation=dilation[0]
    )
    q = _calculate_output_size(
        w, s, stride=stride[1], padding=padding[1], dilation=dilation[1]
    )

    output = torch.empty((n, k, p, q), dtype=input.dtype, device=input.device)

    if bias is None:
        bias = torch.zeros((k,), dtype=output.dtype, device=output.device)

    kernel = _cached_make(
        ntops.kernels.conv2d.premake,
        stride_h=stride[0],
        stride_w=stride[1],
        padding_h=padding[0],
        padding_w=padding[1],
        dilation_h=dilation[0],
        dilation_w=dilation[1],
    )

    kernel(input, weight, bias, output, _get_matmul_input_precision())

    return output
