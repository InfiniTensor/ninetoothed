import math
import torch

import ntops
from ntops.torch.utils import _cached_make


def lp_pool2d(input, norm_type, kernel_size: int | tuple[int, int], stride: None | int | tuple[int, int] = None, ceil_mode=False):
    assert input.ndim == 4 or input.ndim == 3, "Input tensor must be 4-dimensional (N, C, H_in, W_in) or 3-dimensional (C, H_in, W_in)"

    if input.ndim == 3:
        input = input.unsqueeze(0)  # 添加 batch 维度

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if isinstance(stride, int):
        stride = (stride, stride)

    if stride is None:
        stride = kernel_size

    # 计算输出长度
    H_in, W_in = input.shape[-2], input.shape[-1]
    if ceil_mode:
        H_out = math.ceil((H_in - kernel_size[0] + stride[0]) / stride[0])
        W_out = math.ceil((W_in - kernel_size[1] + stride[1]) / stride[1])
    else:
        H_out = math.floor((H_in - kernel_size[0] + stride[0]) / stride[0])
        W_out = math.floor((W_in - kernel_size[1] + stride[1]) / stride[1])
    
    output_shape = (input.shape[0], input.shape[1], H_out, W_out)

    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    block_size = 1024
    if ceil_mode:
        kernel = _cached_make(
            ntops.kernels.lp_pool2d.premake_ceil_mode, 
            input.ndim, 
            kernel_size[0], 
            kernel_size[1],
            stride[0],
            stride[1],
            block_size=block_size,
            ceil_mode=ceil_mode,
            dtype=input.dtype
        )
        kernel(input, output, norm_type, kernel_size[0] * kernel_size[1])
    else:
        kernel = _cached_make(
            ntops.kernels.lp_pool2d.premake, 
            input.ndim, 
            kernel_size[0], 
            kernel_size[1],
            stride[0],
            stride[1],
            block_size=block_size,
            ceil_mode=ceil_mode,
            dtype=input.dtype
        )
        kernel(input, output, norm_type)
    
    
    return output
