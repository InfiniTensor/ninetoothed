import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement

def application(input, values, indices, dim_size, descending):
    val_block = input[0]

    # 初始化结果 buffer
    res_vals = ntl.zeros(val_block.shape, dtype=val_block.dtype)
    res_idxs = ntl.zeros(val_block.shape, dtype=indices.dtype.dtype)
    
    # 用于记录当前写入位置的索引
    output_range = ntl.arange(0, val_block.shape[0])
    # 原始数据的索引，用于比较
    idx_block = ntl.arange(0, val_block.shape[0])

    # 根据排序方向决定处理逻辑
    # argsort 默认是 ascending (从小到大)，即 descending=False
    # 如果 descending=True (largest=True)，我们找最大值
    # 如果 descending=False (largest=False)，我们将值取反后找最大值 (即找最小值)
    if descending:
        working_val = val_block
    else:
        working_val = -val_block

    sentinel = float("-inf")

    # 循环次数为维度的完整大小
    for i in range(dim_size):
        # 找到当前 working_val 中的最大值（及其索引）
        current_max_val = ntl.max(working_val, axis=0)
        current_max_idx = ntl.argmax(working_val, axis=0)

        # 还原真实值（如果是为了找最小值取反过，现在要反回来）
        real_val = -current_max_val if not descending else current_max_val
        real_val = ntl.cast(real_val, res_vals.dtype)

        # 确定当前写入的位置 (target_mask 只有一个位置是 True)
        target_mask = output_range == i
        
        # 将找到的最值和索引写入结果 Tensor
        res_vals = ntl.where(target_mask, real_val, res_vals)
        res_idxs = ntl.where(target_mask, current_max_idx, res_idxs)

        # Mask 掉已经选中的元素，防止下次被选中
        mask_selected = idx_block == current_max_idx
        updated_working_val = ntl.where(mask_selected, sentinel, working_val)
        working_val = ntl.cast(updated_working_val, working_val.dtype)

    # 写回输出
    values[0] = res_vals
    indices[0] = res_idxs


def premake(
    ndim, dim, dim_size, descending, dtype=None, indices_dtype=None, block_size=None
):
    # 使用 reduction 的 arrangement，通常用于处理规约维度的 block 划分
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    # 填充值用于处理 padding (虽然全量 argsort 通常不需要 padding，但为了鲁棒性)
    pad_val = float("-inf") if descending else float("inf")

    tensors = (
        Tensor(ndim, dtype=dtype, other=pad_val), # Input
        Tensor(ndim, dtype=dtype),                # Output Values (辅助，虽然 argsort 主要要 indices)
        Tensor(ndim, dtype=indices_dtype),        # Output Indices
        Tensor(0, constexpr=True, value=dim_size),# Loop bound (k=dim_size)
        Tensor(0, constexpr=True, value=descending),
    )

    return arrangement_, application, tensors