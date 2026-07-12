import torch
import ntops
import ninetoothed
from ntops.torch.utils import _cached_make
# 假设上面的 kernel 代码保存在 ntops.kernels.maximum 中
import ntops.kernels.maximum 

def maximum(input, other, out=None):
    # 1. 处理广播机制 (Broadcasting)
    # 使 input 和 other 具有相同的形状
    input_b, other_b = torch.broadcast_tensors(input, other)
    
    # 2. 确保内存连续 (Contiguous)
    # Triton/DSL kernel 通常假设数据在内存中是紧凑排列的
    input_b = input_b.contiguous()
    other_b = other_b.contiguous()
    
    output_shape = input_b.shape
    
    # 3. 准备输出 Tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        # 简单的形状检查
        assert out.shape == output_shape, f"Output shape mismatch: expected {output_shape}, got {out.shape}"
        if not out.is_contiguous():
            raise RuntimeError("Output tensor must be contiguous for maximum kernel")

    # 4. 设置 Block Size
    # 这里的 1024 是经验值，通常可以根据硬件或 heuristic 动态调整
    block_size = 1024
    
    # 5. 获取并编译 Kernel
    kernel = _cached_make(
        ntops.kernels.maximum.premake, 
        input_b.ndim, 
        input_b.dtype, 
        block_size
    )
    
    # 6. 执行 Kernel
    kernel(input_b, other_b, out)
    
    return out