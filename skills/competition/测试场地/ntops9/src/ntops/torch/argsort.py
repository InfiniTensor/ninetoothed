import torch

import ntops
from ntops.torch.utils import _cached_make

def next_power_of_2(n):
    if n == 0:
        return 1
    return 1 << (n - 1).bit_length()

def get_optimal_block_size(dim_size):
    target_size = next_power_of_2(dim_size)
    if target_size > 1024:
        target_size = 1024
    if target_size < 32:
        target_size = 32
    return target_size

def argsort(input, dim=-1, descending=False, *, out=None):
    """
    Args:
        input: 输入张量
        dim: 排序的维度，默认为最后一维
        descending: 是否降序。默认为 False (升序)，与 torch.argsort 保持一致。
        out: 输出张量 (可选)
    """
    dtype = input.dtype
    indices_dtype = torch.int64

    # 处理 dim
    if dim is None:
        # 如果 dim 为 None，torch 通常会 flatten 后排序，这里为了简化暂不支持或视为 dim=0
        raise NotImplementedError("argsort currently requires a specific dim")
    
    if dim < 0:
        dim += input.ndim
    
    target_dim = dim
    dim_size = input.shape[target_dim]
    
    # 逻辑 Input (无需 flatten，直接作为 ndim tensor 处理，依赖 arrangement 划分)
    input_logic = input
    
    # 计算 Block Size
    block_size = get_optimal_block_size(dim_size)

    # 准备 Output
    # argsort 只需要返回 indices，但我们的 kernel 同时也计算 values (类似 sort)
    # 我们可以分配一个临时的 values tensor
    output_shape = list(input.shape)
    
    if out is not None:
        indices_logic = out
    else:
        indices_logic = torch.empty(
            output_shape, dtype=indices_dtype, device=input.device
        )
    
    # 临时存储排序后的值，Kernel 需要用到它来回写
    values_logic = torch.empty(output_shape, dtype=dtype, device=input.device)

    # 构建 Kernel
    # 注意：这里的 premake 对应上面定义的 ntops.kernels.argsort.premake
    kernel = _cached_make(
        ntops.kernels.argsort.premake,
        input_logic.ndim,
        target_dim,
        dim_size,     # 这里 k = dim_size
        descending,   # argsort 默认 ascending (False)
        dtype,
        indices_dtype,
        block_size,
    )

    # 启动 Kernel
    kernel(input_logic, values_logic, indices_logic, dim_size, descending)

    return indices_logic