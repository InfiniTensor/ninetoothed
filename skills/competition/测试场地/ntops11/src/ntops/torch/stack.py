import torch
import ntops
from ntops.torch.utils import _cached_make

def stack(tensors, dim=0):
    if not tensors:
        raise ValueError("stack expects a non-empty list of tensors")
    
    first_tensor = tensors[0]
    target_dtype = first_tensor.dtype
    target_device = first_tensor.device
    input_shape = first_tensor.shape
    ndim = first_tensor.ndim
    
    # 1. 校验所有输入张量的形状和类型必须一致
    for t in tensors:
        assert t.shape == input_shape, f"Shape mismatch: expected {input_shape}, got {t.shape}"
        assert t.dtype == target_dtype, "Dtype mismatch in input tensors"
        assert t.device == target_device, "Device mismatch in input tensors"

    # 2. 计算输出形状
    # stack 会在 dim 维度插入大小为 len(tensors) 的新维度
    output_shape = list(input_shape)
    output_shape.insert(dim, len(tensors))
    
    # 3. 分配输出内存
    output = torch.empty(output_shape, dtype=target_dtype, device=target_device)
    
    # 4. 准备 Kernel
    # 注意：这里我们使用 input 的 ndim，因为 Kernel 处理的是切片
    block_size = 1024
    kernel = _cached_make(
        ntops.kernels.stack.premake, # 假设上面的代码保存在这里
        ndim,
        block_size=block_size,
        dtype=target_dtype
    )
    
    # 5. 循环执行 (Iterative Execution)
    # 对于每个输入张量，将其拷贝到 output 对应的切片中
    for i, t in enumerate(tensors):
        # output.select(dim, i) 返回的是一个 View（视图），
        # 它共享 output 的内存，但形状与 input 一致。
        # Kernel 就像处理两个普通张量一样处理它们。
        output_slice = output.select(dim, i)
        kernel(t, output_slice)
        
    return output