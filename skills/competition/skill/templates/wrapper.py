"""
Torch 包装层模板
================
提供多种常见包装模式，用于将 kernel 接入 PyTorch 接口
"""

import torch


def flatten_wrapper(kernel_module, input, BLOCK_SIZE=1024):
    """
    Flatten 模式：适用于 element-wise / 激活函数
    将任意 shape 展平后调用 kernel，再恢复原形状
    """
    input_flat = input.flatten()
    output_flat = torch.empty_like(input_flat)
    kernel_module.kernel(input_flat, output_flat, BLOCK_SIZE=BLOCK_SIZE)
    return output_flat.view_as(input)


def direct_wrapper(kernel_module, *args, **kwargs):
    """
    直接模式：适用于 mm, bmm, attention 等
    在包装层创建 output tensor，直接传参
    """
    output = torch.empty(output_shape, dtype=args[0].dtype, device=args[0].device)
    kernel_module.kernel(*args, output, **kwargs)
    return output


def reshape_wrapper(kernel_module, input, *args, BLOCK_SIZE=None):
    """
    Reshape 模式：适用于 rms_norm 等需要 view(-1, last_dim) 的操作
    """
    original_shape = input.shape
    input_2d = input.view(-1, original_shape[-1])
    output_2d = torch.empty_like(input_2d)
    if BLOCK_SIZE is None:
        BLOCK_SIZE = original_shape[-1]
    kernel_module.kernel(input_2d, *args, output_2d, BLOCK_SIZE=BLOCK_SIZE)
    return output_2d.view(original_shape)
