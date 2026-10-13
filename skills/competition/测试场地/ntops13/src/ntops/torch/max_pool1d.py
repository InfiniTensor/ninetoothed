import math
import torch
import torch.nn.functional as F
import ntops
from ntops.torch.utils import _cached_make

def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    # 维度检查
    is_3d = input.ndim == 3
    if input.ndim == 2:
        input = input.unsqueeze(0) # (C, L) -> (1, C, L)
    
    assert input.ndim == 3, "Input tensor must be 3-dimensional (N, C, L) or 2-dimensional (C, L)"

    # 参数标准化
    if stride is None:
        stride = kernel_size
    
    # 处理 Tuple 参数 (虽然 1d 通常是 int，但 torch 允许 (int,))
    if isinstance(kernel_size, tuple): kernel_size = kernel_size[0]
    if isinstance(stride, tuple): stride = stride[0]
    if isinstance(padding, tuple): padding = padding[0]
    if isinstance(dilation, tuple): dilation = dilation[0]

    assert dilation == 1, "Currently only dilation=1 is supported in this DSL implementation"

    # 处理 Explicit Padding
    # 如果有 padding，先用 -inf 填充 input
    if padding > 0:
        input = F.pad(input, (padding, padding), value=float("-inf"))

    L_in = input.shape[-1]

    # 计算输出长度
    if ceil_mode:
        L_out = math.ceil((L_in - kernel_size + stride) / stride)
    else:
        L_out = math.floor((L_in - kernel_size + stride) / stride)

    # 构造 Output
    output_shape = (input.shape[0], input.shape[1], L_out)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    block_size = 1024
    
    kernel = _cached_make(
        ntops.kernels.max_pool1d.premake,
        input.ndim,
        kernel_size,
        stride,
        block_size=block_size,
        ceil_mode=ceil_mode,
        dtype=input.dtype
    )

    kernel(input, output)

    if not is_3d:
        output = output.squeeze(0)

    return output