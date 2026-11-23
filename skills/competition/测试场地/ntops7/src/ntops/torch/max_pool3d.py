import math
import torch
import torch.nn.functional as F
import ntops
from ntops.torch.utils import _cached_make

def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    assert input.ndim == 5, "Input tensor must be 5-dimensional (N, C, D, H, W)"
    assert dilation == 1 or dilation == (1, 1, 1), "Currently only dilation=1 is supported"

    # --- 参数归一化处理 ---
    def _triple(x):
        if isinstance(x, int): return (x, x, x)
        if len(x) == 1: return (x[0], x[0], x[0])
        return x

    k_d, k_h, k_w = _triple(kernel_size)
    
    if stride is None:
        s_d, s_h, s_w = k_d, k_h, k_w
    else:
        s_d, s_h, s_w = _triple(stride)
        
    pad_d, pad_h, pad_w = _triple(padding)

    # --- 精确计算输出尺寸和所需的右侧 Padding ---
    def _calc_dim_and_pad(in_dim, k, s, p, ceil_mode):
        # 1. PyTorch 标准输出尺寸计算公式
        if ceil_mode:
            out_dim = math.ceil((in_dim + 2 * p - k) / s) + 1
        else:
            out_dim = math.floor((in_dim + 2 * p - k) / s) + 1
            
        # 2. PyTorch 特殊边界检查 (Ceil Mode 下排除纯 Padding 的窗口)
        # 逻辑: 如果最后一个窗口的起始位置 >= 原始输入长度 + 左padding，则该窗口无效
        if ceil_mode:
            if (out_dim - 1) * s >= in_dim + p:
                out_dim -= 1
        
        # 3. 反推需要的总长度，以满足 DSL 的 floor tiling
        # 我们需要: (out_dim - 1) * s + k
        needed_len = (out_dim - 1) * s + k
        
        # 4. 计算右侧需要补多少
        # 当前已有: in_dim + p (左侧 padding)
        current_len = in_dim + p
        pad_right = max(0, needed_len - current_len)
        
        return out_dim, pad_right

    D_in, H_in, W_in = input.shape[-3], input.shape[-2], input.shape[-1]

    D_out, pad_r_d = _calc_dim_and_pad(D_in, k_d, s_d, pad_d, ceil_mode)
    H_out, pad_r_h = _calc_dim_and_pad(H_in, k_h, s_h, pad_h, ceil_mode)
    W_out, pad_r_w = _calc_dim_and_pad(W_in, k_w, s_w, pad_w, ceil_mode)

    # --- 应用 Explicit Padding ---
    # F.pad 顺序: (W_left, W_right, H_top, H_bot, D_front, D_back)
    if any(p > 0 for p in [pad_w, pad_r_w, pad_h, pad_r_h, pad_d, pad_r_d]):
        input = F.pad(
            input, 
            (pad_w, pad_r_w, pad_h, pad_r_h, pad_d, pad_r_d), 
            value=float("-inf")
        )

    output_shape = (input.shape[0], input.shape[1], D_out, H_out, W_out)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    block_size = 1024
    
    # 注意: 这里 ceil_mode 永远传 False
    # 因为我们已经通过 Explicit Padding 确保了 floor 切分能得到正确的 output size
    kernel = _cached_make(
        ntops.kernels.max_pool3d.premake,
        input.ndim,
        k_d, k_h, k_w,
        s_d, s_h, s_w,
        block_size=block_size,
        ceil_mode=False, 
        dtype=input.dtype
    )

    kernel(input, output)

    return output