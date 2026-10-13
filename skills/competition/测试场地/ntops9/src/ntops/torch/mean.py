import math
import torch
import ntops
from ntops.torch.utils import _cached_make
# 引入上面定义的 kernel
import ntops.kernels.mean 

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

def mean(
    input,
    dim: int | tuple[int] | list[int] | None = None,
    keepdim=False,
    *,
    dtype=None,
    out=None,
):
    # 1. 确定计算使用的 dtype
    # Mean 操作如果输入是整数，必须提升为浮点数
    if dtype is None:
        if input.dtype.is_floating_point:
            computation_dtype = input.dtype
        else:
            computation_dtype = torch.float32
    else:
        computation_dtype = dtype

    # 2. 计算用于除法的元素总数 N
    if dim is None:
        num_elements = input.numel()
    else:
        if isinstance(dim, int):
            dims = (dim,)
        else:
            dims = tuple(dim)
        
        num_elements = 1
        for d in dims:
            num_elements *= input.shape[d]

    # 3. Kernel 计算 (本质是 Sum，但是使用 computation_dtype)
    
    # --- Case A: Global Mean (所有元素) ---
    if dim is None:
        current = input
        block_size = get_optimal_block_size(current.numel())
        
        # 递归规约 (Global Reduction)
        while current.numel() > 1:
            output_len = math.ceil(current.numel() / block_size)
            output = torch.empty((output_len,), dtype=computation_dtype, device=current.device)
            
            kernel = _cached_make(
                ntops.kernels.mean.premake_all_elements,
                current.ndim,
                computation_dtype, # 确保 kernel 使用浮点
                block_size,
            )
            kernel(current, output)
            current = output
        
        result_sum = current.view(())
        
        # 4. 执行除法
        result = result_sum.div(num_elements)

        if out is not None:
            out.copy_(result)
            return out
        return result

    # --- Case B: Dim Mean (指定维度) ---
    else:
        output_shape = list(input.shape)
        for d in dims:
            if d < 0:
                d += input.ndim
            output_shape[d] = 1
        
        temp_out = torch.empty(output_shape, dtype=computation_dtype, device=input.device)
        block_size = get_optimal_block_size(output_shape[dims[0]])

        kernel = _cached_make(
            ntops.kernels.mean.premake, 
            input.ndim, 
            dims, 
            computation_dtype, # 确保 kernel 使用浮点
            block_size
        )
        kernel(input, temp_out)

        # 4. 执行除法 (In-place 以优化性能)
        temp_out.div_(num_elements)

        if not keepdim:
            dims_to_remove = sorted(
                [d if d >= 0 else d + input.ndim for d in dims], reverse=True
            )
            final_shape = list(output_shape)
            for d in dims_to_remove:
                del final_shape[d]
            
            if not final_shape:
                temp_out = temp_out.view(())
            else:
                temp_out = temp_out.view(final_shape)

        if out is not None:
            out.copy_(temp_out)
            return out

        return temp_out