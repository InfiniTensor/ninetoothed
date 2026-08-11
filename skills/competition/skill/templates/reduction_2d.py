"""
2D 行归约算子模板
=================
适用于：softmax, rms_norm, layer_norm 等
  沿最后一维做归约的操作

模式特征：
- 保留第一维（batch），沿第二维做 tile
- BLOCK_SIZE 通常取 input.shape[-1] 以覆盖整行
- 涉及跨元素归约（sum, max 等）
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

# ============================================================
# Step 1: 定义符号
# ============================================================
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


# ============================================================
# Step 2: 定义 arrangement（数据布局）
# ============================================================
def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((1, BLOCK_SIZE)), output.tile((1, BLOCK_SIZE))


# ============================================================
# Step 3: 定义 application（计算逻辑）
# ============================================================
def application(input, output):
    # 在第二维上做归约
    output = <REDUCTION>  # noqa: F841


# ============================================================
# Step 4: 声明 Tensor 元信息
# ============================================================
# other=float("-inf") 用于给边界外的填充值（如 softmax mask）
tensors = (Tensor(2, other=float("-inf")), Tensor(2))


# ============================================================
# Step 5: 创建 kernel
# ============================================================
kernel = ninetoothed.make(arrangement, application, tensors)
