import torch
import ntops
from ntops.torch.utils import _cached_make

def adaptive_avg_pool2d(input, output_size):
    assert input.ndim == 4 or input.ndim == 3, "Input tensor must be 4-dimensional (N, C, H, W) or 3-dimensional (C, H, W)"
    
    if input.ndim == 3:
        input = input.unsqueeze(0)
        
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    H_in, W_in = input.shape[-2], input.shape[-1]
    H_out, W_out = output_size
    
    # 计算 stride 和 kernel_size (固定窗口策略)
    # 对于 Input % Output == 0 的情况，这与 PyTorch 行为完全一致
    stride_h = H_in // H_out
    stride_w = W_in // W_out
    
    kernel_h = H_in - (H_out - 1) * stride_h
    kernel_w = W_in - (W_out - 1) * stride_w
    
    output_shape = (input.shape[0], input.shape[1], H_out, W_out)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    block_size = 1024
    
    kernel = _cached_make(
        ntops.kernels.adaptive_avg_pool2d.premake,
        input.ndim,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        block_size=block_size,
        dtype=input.dtype
    )
    
    # 传入实际的窗口面积作为除数
    area = kernel_h * kernel_w
    kernel(input, output, area)
    
    return output