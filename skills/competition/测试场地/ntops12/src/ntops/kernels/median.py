import functools
import triton.language as tl  # [FIX 1] 导入标准 triton language
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ntops.kernels.reduction import arrangement

def application(input, values, indices, loop_k):
    val_block = input[0]

    # 初始化用于查找的 working_val
    working_val = -val_block
    sentinel = float("-inf")
    
    # 原始索引
    idx_block = ntl.arange(0, val_block.shape[0])

    # 初始化结果 buffer (Scalar)
    final_val = ntl.zeros([], dtype=val_block.dtype)
    
    # [FIX 2] 使用 tl.int32 而不是字符串 "int32"
    final_idx = ntl.zeros([], dtype=tl.int32) 

    # 循环
    for i in range(loop_k + 1):
        current_max_val = ntl.max(working_val, axis=0)
        current_max_idx = ntl.argmax(working_val, axis=0) # 返回 int32

        if i == loop_k:
            real_val = -current_max_val
            final_val = ntl.cast(real_val, values.dtype.dtype)
            final_idx = current_max_idx # int32 -> int32

        # Mask 逻辑
        mask_selected = idx_block == current_max_idx
        updated_working_val = ntl.where(mask_selected, sentinel, working_val)
        working_val = ntl.cast(updated_working_val, working_val.dtype)

    # 写回输出
    values[0] = final_val
    # Cast int32 -> int64 (或输出需要的类型)
    indices[0] = ntl.cast(final_idx, indices.dtype.dtype)

def premake(
    ndim, dim, loop_k, dtype=None, indices_dtype=None, block_size=None
):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)
    pad_val = float("inf")

    tensors = (
        Tensor(ndim, dtype=dtype, other=pad_val),     # Input
        Tensor(ndim, dtype=dtype),                    # Output Values
        Tensor(ndim, dtype=indices_dtype),            # Output Indices
        Tensor(0, constexpr=True, value=loop_k),      # Loop bound
    )

    return arrangement_, application, tensors