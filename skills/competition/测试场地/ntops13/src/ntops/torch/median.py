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

def median(input, dim=-1, keepdim=False, *, out=None):
    """
    Args:
        input: 输入张量
        dim: 计算中位数的维度
        keepdim: 是否保持输出维度
        out: (values, indices) 元组 (可选)
    Returns:
        (values, indices) namedtuple or tuple
    """
    dtype = input.dtype
    indices_dtype = torch.int64

    # 处理 dim
    if dim is None:
        raise NotImplementedError("median currently requires a specific dim")
    
    if dim < 0:
        dim += input.ndim
    
    target_dim = dim
    dim_size = input.shape[target_dim]
    
    # 计算中位数在排序后的索引位置 (PyTorch 默认行为：(N-1) // 2)
    # 比如 N=3, idx=1; N=4, idx=1
    median_rank = (dim_size - 1) // 2

    input_logic = input
    block_size = get_optimal_block_size(dim_size)

    # 准备 Output
    # Median 操作会 reduce 掉 target_dim，或者 keepdim=True
    # 但 Kernel 这里的 arrangement 是一对一的 (Input Block -> Output Block)
    # 通常 reduction kernel 输出的 shape 在 target_dim 上是 1 或被 squeeze
    # 这里我们按照 keepdim=True 的形状申请，最后根据参数 squeeze
    
    output_shape = list(input.shape)
    output_shape[target_dim] = 1 # 结果只有 1 个元素
    
    if out is not None:
        values_logic, indices_logic = out
    else:
        values_logic = torch.empty(output_shape, dtype=dtype, device=input.device)
        indices_logic = torch.empty(output_shape, dtype=indices_dtype, device=input.device)

    # 构建 Kernel
    # 注意：我们将 median_rank 作为 loop_k 传入 premake
    kernel = _cached_make(
        ntops.kernels.median.premake, # 假设上面的 kernel 代码在这个路径
        input_logic.ndim,
        target_dim,
        median_rank,   # loop_k
        dtype,
        indices_dtype,
        block_size,
    )

    # 启动 Kernel
    kernel(input_logic, values_logic, indices_logic, median_rank)

    # 处理 keepdim
    if not keepdim:
        values_logic = values_logic.squeeze(dim)
        indices_logic = indices_logic.squeeze(dim)

    return torch.return_types.median((values_logic, indices_logic))