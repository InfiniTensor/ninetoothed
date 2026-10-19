"""
激活函数算子模板
================
适用于：silu, swiglu, gelu 等激活函数

模式特征：
- element-wise，沿最后一维分块，保留 strides（支持非连续张量）
- 使用 ntl.sigmoid, ntl.cast 等 ntl 语言 API
- 通常涉及类型提升到 float32 再计算
- 使用 _element_wise_arrangement 通用布局替代 1D tile
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

# ============================================================
# Step 1: 定义符号
# ============================================================
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


# ============================================================
# Step 2: 通用 arrangement（保留 strides，支持非连续张量）
# ============================================================
def _element_wise_arrangement(*tensors, block_size):
    ndim = max(tensor.ndim for tensor in tensors)
    assert all(tensor.ndim == ndim or tensor.ndim == 0 for tensor in tensors)
    tile_shape = tuple(1 for _ in range(ndim - 1)) + (block_size,)
    return tuple(
        tensor.tile(tile_shape) if tensor.ndim != 0 else tensor
        for tensor in tensors
    )


def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return _element_wise_arrangement(input, output, block_size=BLOCK_SIZE)


# ============================================================
# Step 3: 定义 application
# ============================================================
def application(input, output):
    input_loaded = input
    # 类型提升到 float32 进行计算
    #
    # GELU approximate: 0.5 * x * (1 + tanh(0.79788 * (x + 0.044715 * x^3)))
    #   x**3 → x*x*x, tanh → (exp(t)-exp(-t))/(exp(t)+exp(-t))
    # GELU exact:      x * 0.5 * (1 + erf(x / sqrt(2)))
    # Silu:            x * sigmoid(x)
    # 注意：`**` 运算符不可用（Triton tensor 无 __pow__）
    # 注意：模块级变量引用被 AST 原样嵌入 → NameError，用字面量
    output = <FORMULA>  # noqa: F841


# ============================================================
# Step 4: 声明 Tensor 元信息（ndim 会展开为具体值）
# ============================================================
tensors = (Tensor(1), Tensor(1))


# ============================================================
# Step 5: 创建 kernel
# ============================================================
kernel = ninetoothed.make(arrangement, application, tensors)
