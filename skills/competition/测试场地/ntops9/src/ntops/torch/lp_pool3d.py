import math
import torch

import ntops
from ntops.torch.utils import _cached_make


def lp_pool3d(input, norm_type, kernel_size: int | tuple[int, int, int], stride: None | int | tuple[int, int, int] = None, ceil_mode=False):
    assert input.ndim == 5 or input.ndim == 4, "Input tensor must be 4-dimensional (N, C, D_in, H_in, W_in) or 3-dimensional (C, D_in, H_in, W_in)"

    if input.ndim == 4:
        input = input.unsqueeze(0)  # 添加 batch 维度

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)

    if isinstance(stride, int):
        stride = (stride, stride, stride)

    if stride is None:
        stride = kernel_size

    # 计算输出长度
    N, C, D_in, H_in, W_in = input.shape
    if ceil_mode:
        D_out = math.ceil((D_in - kernel_size[0] + stride[0]) / stride[0])
        H_out = math.ceil((H_in - kernel_size[1] + stride[1]) / stride[1])
        W_out = math.ceil((W_in - kernel_size[2] + stride[2]) / stride[2])
    else:
        D_out = math.floor((D_in - kernel_size[0] + stride[0]) / stride[0])
        H_out = math.floor((H_in - kernel_size[1] + stride[1]) / stride[1])
        W_out = math.floor((W_in - kernel_size[2] + stride[2]) / stride[2])
    
    output_shape = (N, C, D_out, H_out, W_out)

    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    block_size = 256
    if ceil_mode:
        kernel = _cached_make(
            ntops.kernels.lp_pool3d.premake_ceil_mode, 
            input.ndim, 
            kernel_size[0],
            kernel_size[1],
            kernel_size[2],
            stride[0],
            stride[1],
            stride[2],
            block_size=block_size,
            ceil_mode=ceil_mode,
            dtype=input.dtype,
        )
        kernel(input, output, norm_type, kernel_size[0] * kernel_size[1] * kernel_size[2])
    else:
        kernel = _cached_make(
            ntops.kernels.lp_pool3d.premake, 
            input.ndim, 
            kernel_size[0], 
            kernel_size[1],
            kernel_size[2],
            stride[0],
            stride[1],
            stride[2],
            block_size=block_size,
            ceil_mode=ceil_mode,
            dtype=input.dtype
        )
        kernel(input, output, norm_type)
    
    
    return output
