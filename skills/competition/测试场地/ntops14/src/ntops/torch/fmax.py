import torch
import ntops
import ninetoothed
from ntops.torch.utils import _cached_make

def fmax(input, other, out=None):
    input_b, other_b = torch.broadcast_tensors(input, other)
    
    # 必须 contiguous，因为 kernel 使用 flatten() 视作 1D 数组处理
    input_b = input_b.contiguous()
    other_b = other_b.contiguous()
    
    output_shape = input_b.shape
    
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == output_shape
        if not out.is_contiguous():
             # 如果用户传入的 out 不连续，会导致写入错误，这里最好报错或处理
             # 但为了性能，通常假设用户知道自己在做什么，或者在这里强制 check
             raise RuntimeError("Output tensor must be contiguous for fmax kernel")

    block_size = 1024
    
    # 注意：这里传递 input_b.ndim 给 premake
    # 虽然我们在 kernel 里 flatten 了，但 Tensor(ndim) 的定义有助于 ninetoothed 校验参数
    kernel = _cached_make(
        ntops.kernels.fmax.premake, 
        input_b.ndim, 
        input_b.dtype, 
        block_size
    )
    
    kernel(input_b, other_b, out)
    return out