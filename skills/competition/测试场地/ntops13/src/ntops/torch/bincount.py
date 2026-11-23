import torch
import ntops
from ntops.torch.utils import _cached_make

def bincount(input, weights=None, minlength=0):
    if input.ndim != 1:
        raise ValueError("input must be 1-dimensional")
    if weights is not None and weights.ndim != 1:
        raise ValueError("weights must be 1-dimensional")
    if weights is not None and weights.shape[0] != input.shape[0]:
        raise ValueError("weights should have the same shape as input")

    T = input.shape[0]
    
    # 计算输出大小 S
    if T > 0:
        max_val = input.max().item()
        S = max(int(max_val) + 1, minlength)
    else:
        S = minlength
    
    # 计算 T 的下一个 2 的幂，用于 arange 和 tiling
    T_pow2 = 1 << (T - 1).bit_length() if T > 0 else 1
    
    # S 不需要严格的 2 的幂用于 Kernel 逻辑，但作为参数传递保持一致性
    S_pow2 = 1 

    # 处理 Weights 和 Output 类型
    # Torch 语义: weights 为 None 时返回 Long (int64), 否则返回 weights 的类型
    if weights is None:
        weights = torch.ones_like(input, dtype=torch.int64)
        out_dtype = torch.int64
    else:
        out_dtype = weights.dtype

    # 准备 Output Tensor
    output = torch.zeros(S, dtype=out_dtype, device=input.device)
    
    # 准备 Bin IDs (辅助 Tensor，用于告知每个 Block 它负责哪些 Bin)
    bin_ids = torch.arange(S, dtype=input.dtype, device=input.device)

    # Block Size: 决定每个 Grid 处理多少个 Output Bin
    block_size = 128 

    kernel = _cached_make(ntops.kernels.bincount.premake, 
                          T_pow2=T_pow2, 
                          S_pow2=S_pow2, 
                          dtype=out_dtype, 
                          block_size=block_size)
    
    kernel(input, weights, output, bin_ids, T, S)

    return output