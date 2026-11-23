"""
2D Matmul 算子模板
==================
适用于：mm, addmm 等矩阵乘法操作

模式特征：
- 使用 3 个 block_size 符号（M, N, K）由 autotune 自动搜索
- output 按 (BLOCK_SIZE_M, BLOCK_SIZE_N) 分块
- input 和 other 通过 tile + expand + squeeze 对齐到 output 分块
- application 中使用循环 dot 累加
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

# ============================================================
# Step 1: 定义符号（meta 类型，由 autotune 自动搜索）
# ============================================================
BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()
BLOCK_SIZE_K = block_size()


# ============================================================
# Step 2: 定义 arrangement（数据布局）
# ============================================================
def arrangement(
    input,
    other,
    output,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    output_arranged = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    # input: (M, K) -> (BLOCK_SIZE_M, BLOCK_SIZE_K) -> tile(1, -1) -> expand(-1, N_blocks)
    input_arranged = input.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    # other: (K, N) -> (BLOCK_SIZE_K, BLOCK_SIZE_N) -> tile(-1, 1) -> expand(M_blocks, -1)
    other_arranged = other.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
    other_arranged = other_arranged.tile((-1, 1))
    other_arranged = other_arranged.expand((output_arranged.shape[0], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze(1)

    return input_arranged, other_arranged, output_arranged


# ============================================================
# Step 3: 定义 application（计算逻辑）
# ============================================================
def application(input, other, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(input.shape[0]):
        accumulator += ntl.dot(input[k], other[k])
    output = accumulator


# ============================================================
# Step 4: 声明 Tensor 元信息
# ============================================================
tensors = (Tensor(2), Tensor(2), Tensor(2))


# ============================================================
# Step 5: 创建 kernel
# ============================================================
kernel = ninetoothed.make(arrangement, application, tensors)
