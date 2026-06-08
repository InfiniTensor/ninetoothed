"""
Element-wise 算子模板
======================
适用于：add, mul, relu, gelu, silu 等逐元素操作

模式特征：
- 所有张量沿最后一维均匀分块，无跨块归约
- 支持非连续张量（transpose、slice 等），保留原始 strides
- BLOCK_SIZE 作为编译时常量传入
- 推荐使用工厂函数模式（make_*）以支持动态 ndim

用法示例：
    add_kernel = make_add(ndim=2)
    add_kernel(x, y, out, BLOCK_SIZE=1024)

    relu_kernel = make_relu(ndim=1)
    relu_kernel(x, out, BLOCK_SIZE=1024)

# ⚠️ AST 跟踪约束（重要）
# ============================================================
# application() 内的 Python 代码会通过 AST 跟踪直接嵌入
# 生成的 Triton 代码。Triton 的编译环境没有标准 Python 库，
# 因此必须遵守以下规则：
#
# ❌ 禁止：math.*、torch.*、numpy.*
# ❌ 禁止：模块级变量引用（原样嵌入 → NameError）
# ❌ 禁止：** 运算符（Triton tensor 无 __pow__）
# ✅ 允许：ntl.* 函数、字面量数值、四则运算
# ✅ 推荐：x * x * x 代替 x ** 3
# ✅ 推荐：0.7978845608028654 代替 math.sqrt(2.0 / math.pi)
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor


# ============================================================
# Step 1: 定义符号（编译时常量）
# ============================================================
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


# ============================================================
# Step 2: 通用 arrangement（数据布局）
# ============================================================
def _element_wise_arrangement(*tensors, block_size):
    """通用 element-wise arrangement：保留 strides，支持非连续张量。

    工作原理：
    - 自动确定所有张量的最大 ndim
    - 0 维张量（标量）原样传递，不做 tile
    - 高维张量构造 tile_shape = (1, ..., 1, block_size)，
      前 ndim-1 维为 1 不做 tile，只在最后一维分块。
      这样高维张量的行/列 strides 被保留，Triton 能够
      通过 ptr + row * stride_row + col * stride_col 正确寻址。

    注意：标量广播需要通过 expand_as 创建 stride=0 视图调用方完成。
    """
    ndim = max(tensor.ndim for tensor in tensors)
    assert all(tensor.ndim == ndim or tensor.ndim == 0 for tensor in tensors)

    tile_shape = tuple(1 for _ in range(ndim - 1)) + (block_size,)

    return tuple(
        tensor.tile(tile_shape) if tensor.ndim != 0 else tensor
        for tensor in tensors
    )


# ============================================================
# Step 3: 定义 application（计算逻辑）
# ============================================================
def application(input, output):
    output = <FORMULA>  # noqa: F841


# ============================================================
# Step 4: 声明 Tensor 元信息
# ============================================================
# Tensor(1) 表示 1 维张量 这里的tensors相当于申请了两个一维张量
tensors = (Tensor(1), Tensor(1))


# ============================================================
# Step 5: 创建 kernel
# ============================================================
kernel = ninetoothed.make(arrangement, application, tensors)
