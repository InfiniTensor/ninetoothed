import functools
from ninetoothed import Tensor
import ninetoothed.language as ntl

def arrangement(input, weights, output, bin_ids, T, S, T_pow2, S_pow2, block_size=None):
    # input: (T,)
    # weights: (T,)
    # output: (S,) - 这是我们要写入的结果
    # bin_ids: (S,) - 这是辅助索引，对应 output 的每个位置

    # 1. 对 Output 进行分块并行 (Grid 维度)
    # output_tiled: (GridSize, block_size)
    output_tiled = output.tile((block_size,))
    
    # bin_ids 也随 output 一起分块，以便我们在 Kernel 中知道当前处理的是哪些 bin
    bin_ids_tiled = bin_ids.tile((block_size,))

    # 2. Input 和 Weights 需要被所有 Block 访问
    # 我们先将它们扩展到 GridSize，然后调整维度以符合 T_pow2 的要求
    grid_size = output_tiled.shape[0]
    
    # input: (T,) -> (GridSize, T) -> (GridSize, 1, T_pow2) (通过 tile dim 1)
    # 注意：这里的 tile((1, T_pow2)) 是为了让 ninetoothed 框架正确处理内部维度
    input_expand = input.unsqueeze(0).expand((grid_size, -1))
    input_tiled = input_expand.tile((1, T_pow2)).squeeze(1)

    weights_expand = weights.unsqueeze(0).expand((grid_size, -1))
    weights_tiled = weights_expand.tile((1, T_pow2)).squeeze(1)

    return input_tiled, weights_tiled, output_tiled, bin_ids_tiled, T, S

def application(input, weights, output, bin_ids, T, S):
    # input: (1, T_pow2)  <-- 来自 arrangement 的广播
    # weights: (1, T_pow2)
    # output: (block_size,) <-- 当前 block 负责的输出片段
    # bin_ids: (block_size,) <-- 当前 block 负责的 bin 索引

    # 1. 准备维度以便广播比较
    # 我们要计算矩阵: Match[i, j] = (bin_ids[i] == input[j])
    # bin_ids: (block_size, 1)
    # input:   (1, T_pow2)
    
    bin_ids_col = ntl.expand_dims(bin_ids, 1)       # (block_size, 1)
    
    # 确保 input 和 weights 在 dim 0 上广播以匹配 block_size
    input_b = ntl.broadcast_to(input, (bin_ids.shape[0], input.shape[1]))     # (block_size, T_pow2)
    weights_b = ntl.broadcast_to(weights, (bin_ids.shape[0], weights.shape[1])) # (block_size, T_pow2)

    # 2. 生成有效性掩码 (处理 Padding)
    # input 的真实长度是 T，T_pow2 之外的是 padding
    col_indices = ntl.arange(0, input.shape[1])     # (T_pow2,)
    col_valid = col_indices < T                     # (T_pow2,)
    
    # 广播 mask
    col_valid_b = ntl.expand_dims(col_valid, 0)
    col_valid_b = ntl.broadcast_to(col_valid_b, (bin_ids.shape[0], input.shape[1]))

    # 3. 核心计算：Masking + Sum
    # 找出哪些 input 值落入了当前 block 负责的 bin 中
    match_mask = (input_b == bin_ids_col)
    
    # 结合有效性检查
    final_mask = ntl.where(col_valid_b, match_mask, False)
    
    # 选择权重 (如果 weights 是全 1，则相当于计数)
    selected = ntl.where(final_mask, weights_b, 0.0)
    
    # 沿着 T 维度求和，得到每个 bin 的总值
    result = ntl.sum(selected, axis=1) # (block_size,)

    # 4. 写回
    output = result

def premake(T_pow2, S_pow2, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, T_pow2=T_pow2, S_pow2=S_pow2, block_size=block_size)

    tensors = (
        Tensor(1, dtype=int, shape_options={'constexpr': True}),    # input
        Tensor(1, dtype=dtype, shape_options={'constexpr': True}),  # weights
        Tensor(1, dtype=dtype, shape_options={'constexpr': True}),  # output
        Tensor(1, dtype=int, shape_options={'constexpr': True}),    # bin_ids
        Tensor(0, dtype=int, constexpr=True), # T
        Tensor(0, dtype=int, constexpr=True), # S
    )

    return arrangement_, application, tensors