import math
import torch

import ntops
from ntops.torch.utils import _cached_make


def lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    """
    一维 Lp 池化

    参数:
        input: (N, C, L_in) 输入张量
        norm_type: Lp 范数的 p 值（1.0, 2.0, 等）
        kernel_size: 窗口大小
        stride: 步长，默认等于 kernel_size
        ceil_mode: 是否使用 ceil 模式计算输出长度

    返回:
        output: (N, C, L_out) 输出张量
    """

    assert input.ndim == 3 or input.ndim == 2, (
        "Input tensor must be 3-dimensional (N, C, L_in) or (C, L_in)"
    )
    if input.ndim == 2:
        input = input.view(1, input.shape[0], input.shape[1])

    if stride is None:
        stride = kernel_size

    L_in = input.shape[-1]

    # 计算输出长度
    if ceil_mode:
        L_out = math.ceil((L_in - kernel_size + stride) / stride)
    else:
        L_out = math.floor((L_in - kernel_size + stride) / stride)

    output_shape = (input.shape[0], input.shape[1], L_out)

    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    block_size = 1024
    kernel = _cached_make(
        ntops.kernels.lp_pool1d.premake,
        input.ndim,
        kernel_size,
        stride,
        ceil_mode=ceil_mode,
        dtype=input.dtype,
        block_size=block_size
    )

    kernel(input, output, norm_type, kernel_size)

    return output
