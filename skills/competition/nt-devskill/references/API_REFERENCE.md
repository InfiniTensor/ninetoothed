# NineToothed API Reference

## Core API

| API                | Signature / Usage                                     | Description                                                                        |
| ------------------ | ----------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `ninetoothed.make` | `ninetoothed.make(arrangement, application, tensors)` | Compiles an arrangement and application function into a callable kernel.           |
| `ninetoothed.jit`  | `@ninetoothed.jit`                                    | Decorator for JIT-compiling a Triton-like kernel function (alternative to `make`). |

### Tensor & Shape Description

| API                                                 | Description                                                                                                |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `Tensor(ndim)`                                      | Declares a tensor with `ndim` dimensions for the kernel.                                                   |
| `Tensor(ndim, other=fill_value)`                    | Declares a tensor with a fill value for out-of-bounds positions (e.g., `float("-inf")` for max stability). |
| `Tensor(ndim, shape_options={...})`                 | With shape constraints (e.g., `{"constexpr": True, "upper_bound": 128}`).                                  |
| `Tensor(ndim, shape_options=(opts_0, opts_1, ...))` | Per-dimension shape options as a tuple (e.g., for attention with `head_dim` constraint).                   |
| `Tensor(0)`                                         | A scalar value (used for `eps`, `scale`, `beta`, `alpha`, etc.).                                           |

### Symbol & block_size

| API                                           | Description                                                                        |
| --------------------------------------------- | ---------------------------------------------------------------------------------- |
| `Symbol(name, constexpr=True)`                | A compile-time constant symbol. Used with `tile()` for block sizes.                |
| `Symbol(name, constexpr=True, upper_bound=N)` | A compile-time constant with an upper bound hint (helps compiler optimize).        |
| `Symbol(name, meta=True)`                     | A meta-symbol for autotuning. The autotuner searches over its values.              |
| `Symbol(name, meta=True, upper_bound=N)`      | A meta-symbol with bounded search range.                                           |
| `block_size()`                                | A shortcut for `Symbol(..., meta=True)` – creates an auto-tunable block dimension. |

### Tile / Arrangement Primitives

| API                                                        | Description                                                                                    |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `tensor.tile(shape)`                                       | Splits tensor dimensions into tiles. `-1` means "infer from the original extent".              |
| `tensor.tile(shape, strides=...)`                          | Tile with custom strides (e.g., `(-1, -1, -1, 1)` for non-contiguous access).                  |
| `tensor.tile(shape, dilation=...)`                         | Tile with dilation (e.g., `(1, 1, 1, 2)` for interleaved access patterns).                     |
| `arranged.tile((1, -1))`                                   | Further tiling inside a previously tiled dimension.                                            |
| `arranged.expand(shape)`                                   | Repeats the tile to cover more program instances (broadcasting).                               |
| `arranged.dtype = arranged.dtype.squeeze(dim)`             | Removes a tiled dimension to recover the original rank. Supports tuple dim: `squeeze((0, 1))`. |
| `arranged.dtype.dtype = arranged.dtype.dtype.squeeze(dim)` | Second-level dtype access for high-dimensional (4D+) tiling (e.g., bmm, attention).            |
| `arranged.ravel()`                                         | Flattens all tile dimensions into a single dimension.                                          |
| `arranged.flatten(end_dim=N, start_dim=M)`                 | Partially flattens between specified dimension indices.                                        |
| `arranged.permute(dims)`                                   | Permutes tile dimensions (e.g., `(1, 0)` for transpose).                                       |
| `arranged.squeeze(dim)`                                    | Removes a dimension of size 1 from the arranged tensor.                                        |

### ninetoothed.language (ntl)

#### 创建与填充

| API                             | Description                                 |
| ------------------------------- | ------------------------------------------- |
| `ntl.zeros(shape, dtype)`       | Creates a zero-filled local tensor.         |
| `ntl.full(shape, value, dtype)` | Creates a local tensor filled with `value`. |

#### 算术与激活

| API                          | Description                                                                                         |
| ---------------------------- | --------------------------------------------------------------------------------------------------- |
| `ntl.dot(a, b)`              | Matrix multiplication on tiles.                                                                     |
| `+`, `-`, `*`, `/`           | Element-wise arithmetic operators.                                                                  |
| `ntl.exp(x)`                 | Element-wise exponential (base e).                                                                  |
| `ntl.exp2(x)`                | Element-wise base-2 exponential (faster than `exp`; multiply input by `1.44269504089` = `log2(e)`). |
| `ntl.sqrt(x)`                | Element-wise square root.                                                                           |
| `ntl.rsqrt(x)`               | Element-wise reciprocal square root (`1/sqrt(x)`, more efficient than `1/ntl.sqrt(x)`).             |
| `ntl.sigmoid(x)`             | Element-wise sigmoid function.                                                                      |
| `ntl.maximum(a, b)`          | Element-wise max of two tensors.                                                                    |
| `ntl.minimum(a, b)`          | Element-wise min of two tensors.                                                                    |
| `ntl.where(condition, a, b)` | Element-wise conditional selection.                                                                 |

#### 归约

| API                 | Description                                                                         |
| ------------------- | ----------------------------------------------------------------------------------- |
| `ntl.sum(x, axis)`  | Reduces by summation along an axis. `ntl.sum(x)` without axis reduces all elements. |
| `ntl.max(x, axis)`  | Reduces by max along an axis. `ntl.max(x)` without axis reduces all elements.       |
| `ntl.mean(x, axis)` | Mean reduction along an axis.                                                       |

#### 类型与转置

| API                  | Description                                                                        |
| -------------------- | ---------------------------------------------------------------------------------- |
| `ntl.cast(x, dtype)` | Cast tensor to a different dtype (e.g., `ntl.cast(x, ntl.float32)` for precision). |
| `tensor.to(dtype)`   | Convert tensor to a different dtype (alternative to `cast`).                       |
| `ntl.trans(x)`       | Transpose the last two dimensions.                                                 |

#### 边界检查

| API                   | Description                                                                                                                                                |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tensor.offsets(dim)` | Returns a **1D array** of the global offsets of the current tile along `dim`. For 2D+ comparisons, you **must** broadcast with `[:, None]` or `[None, :]`. |
| `tensor.source.shape` | Access the original (untiled) tensor shape (used to construct mask conditions).                                                                            |

#### `.offsets()` Broadcasting Rules (CRITICAL)

`.offsets(dim)` always returns a **1D array** matching the tile's extent along that dimension. When comparing offsets from different dimensions in a 2D tile, you must broadcast them into 2D:

```python
# ❌ WRONG — both are 1D, comparison produces wrong 1D result
mask = output.offsets(0) == output.offsets(1)

# ✅ CORRECT — broadcast to 2D
row_idx = output.offsets(0)[:, None]   # shape: [BLOCK_M, 1]
col_idx = output.offsets(1)[None, :]   # shape: [1, BLOCK_N]
mask = row_idx == col_idx              # shape: [BLOCK_M, BLOCK_N]
```

Common patterns:
```python
# Boundary check (e.g., attention masking)
k_pos = k[i].offsets(-2)[:, None]                      # [BLOCK_K, 1]
valid = k_pos < k.source.shape[-2]                      # broadcast OK
qk = ntl.where(valid, qk, float("-inf"))

# Diagonal check (e.g., eye matrix)
row = output.offsets(0)[:, None]                        # [BLOCK_M, 1]
col = output.offsets(1)[None, :]                        # [1, BLOCK_N]
output = ntl.where(row == col, 1.0, 0.0)               # [BLOCK_M, BLOCK_N]
```

> **Warning:** Forgetting `[:, None]` is the #1 cause of silent correctness errors in 2D+ kernels. The kernel will compile and run, but produce wrong results.

### libdevice 数学函数库（扩展数学函数）

`ntl` 直接暴露的数学函数有限（exp, exp2, sqrt, rsqrt, sigmoid, maximum, minimum）。对于**其他数学函数**（lgamma, log, log2, tgamma, erf, pow, floor, ceil, tanh, cos, sin, abs 等），需要通过 Triton 的 `libdevice` 模块访问。

#### 正确的导入和使用方式（CRITICAL）

```python
# ✅ CORRECT: 模块级全局导入 libdevice
from triton.language.extra import libdevice
import ninetoothed.language as ntl
from ninetoothed import Tensor

def application(input, output):
    # 使用 libdevice.lgamma, libdevice.log, etc.
    output = libdevice.lgamma(ntl.cast(input, ntl.float32))

# ❌ WRONG: 通过 ntl.libdevice 访问（会在 MetaX 上编译失败）
import ninetoothed.language as ntl

def application(input, output):
    output = ntl.libdevice.lgamma(input)  # MetaX: "module 'triton.language' has no attribute 'libdevice'"
```

**为什么 `ntl.libdevice` 不行：** ninetoothed 的编译器将 `ntl.X` 映射为 `triton.language.X`。但 `triton.language.libdevice` 不存在（MetaX Triton 中 libdevice 在 `triton.language.extra.libdevice`）。正确的做法是将 `libdevice` 作为模块级全局变量导入，ninetoothed 的 `_Inliner` 会检测到它并自动插入正确的 import。

#### libdevice 常用函数列表

| 函数                       | 签名                  | 说明                     |
| -------------------------- | --------------------- | ------------------------ |
| `libdevice.lgamma(x)`      | float → float         | log(                     | Γ(x) | ) 对数伽马函数 |
| `libdevice.log(x)`         | float → float         | 自然对数                 |
| `libdevice.log2(x)`        | float → float         | 以 2 为底的对数          |
| `libdevice.exp(x)`         | float → float         | 指数函数                 |
| `libdevice.tgamma(x)`      | float → float         | 伽马函数 Γ(x)            |
| `libdevice.erf(x)`         | float → float         | 误差函数                 |
| `libdevice.pow(x, y)`      | float × float → float | 幂函数                   |
| `libdevice.floor(x)`       | float → float         | 向下取整                 |
| `libdevice.ceil(x)`        | float → float         | 向上取整                 |
| `libdevice.tanh(x)`        | float → float         | 双曲正切                 |
| `libdevice.cosh(x)`        | float → float         | 双曲余弦                 |
| `libdevice.sinh(x)`        | float → float         | 双曲正弦                 |
| `libdevice.copysign(x, y)` | float × float → float | 取 x 的绝对值 + y 的符号 |

#### libdevice vs ntl 组合：选用决策指南

**规则 1：优先使用 ntl 原生函数**（当 ntl 已暴露时）

```python
# ✅ ntl 已有 → 直接用
output = ntl.exp(x)        # 不要用 libdevice.exp
output = ntl.sqrt(x)       # 不要用 libdevice.sqrt
output = ntl.sigmoid(x)    # 不要用 libdevice 组合

# ❌ 不要用 libdevice 重新实现 ntl 已有的函数
output = libdevice.exp(x)  # 多余，且可能编译不同
```

**规则 2：必须使用 libdevice**（当 ntl 未暴露，且 libdevice 有直接对应时）

```python
# libdevice 有直接对应函数，不要用 ntl 组合重新实现
output = libdevice.lgamma(ntl.cast(x, ntl.float32))  # ✅
output = libdevice.log(ntl.cast(x, ntl.float32))     # ✅
output = libdevice.copysign(ntl.cast(x, ntl.float32), ntl.cast(y, ntl.float32))  # ✅

# ❌ 不要用 ntl 组合重新实现 libdevice 已有的函数
output = ntl.where(y >= 0, ntl.abs(x), -ntl.abs(x))  # copysign 错误！不处理 -0.0
output = ntl.log(ntl.exp(x) + 1)  # softplus 可以，但精度不如 libdevice
```

**规则 3：ntl 组合**（当两者都没有时）

```python
# libdevice 和 ntl 都没有 → 用 ntl 基础运算组合
output = x * ntl.sigmoid(x)                                    # SiLU
output = a * (b * ntl.sigmoid(ntl.cast(b, ntl.float32)))       # SwiGLU
output = ntl.maximum(x, 0) + ntl.log(1 + ntl.exp(-ntl.abs(x))) # softplus
```

**规则 4：libdevice 只支持 float32/float64**（CRITICAL）

```python
# ✅ 必须显式 cast 到 float32
x_f32 = ntl.cast(x, ntl.float32)
y_f32 = ntl.cast(y, ntl.float32)
output = libdevice.copysign(x_f32, y_f32)  # 结果自动 cast 回 input dtype

# ❌ 不 cast → float16 输入会产生错误结果
output = libdevice.copysign(x, y)  # x,y 是 float16 时行为未定义
```

#### GPU 侧常量与精度控制

**在 kernel 中使用数学常量时，必须确保常量在 GPU fp32 精度下计算：**

```python
# ✅ GPU 侧常量：用 ntl.cast 确保 GPU fp32 精度
PI = ntl.cast(3.141592653589793, ntl.float32)
output = input * ntl.cast(180.0, ntl.float32) / PI

# ⚠️ Python 侧常量：可能有效，但不够显式
import math
_RAD_TO_DEG = 180.0 / math.pi  # Python float64
output = input * _RAD_TO_DEG   # 依赖编译器对 Python float 的处理

# ✅ 推荐模式：显式 cast 所有常量
def application(input, output):
    scale = ntl.cast(180.0, ntl.float32) / ntl.cast(3.141592653589793, ntl.float32)
    output = input * scale
```

> **原则：** kernel 内部的所有数值运算都应在 GPU 精度下进行。Python float 常量会被编译器提升，但显式 `ntl.cast` 更安全、可读性更好，且避免跨平台精度差异。

### Cross-Tensor Gather Indexing（跨 tensor 索引限制）

九齿的 `tensor[idx]`（gather 语义）在底层使用**第一个 tensor 的 `size_1`**（最后一维大小）做索引分解：

```
row = idx // first_tensor.size_1
col = idx % first_tensor.size_1
physical_offset = row * tensor.size_1 + col
```

**当跨 tensor 索引时**（如用 source 的 offsets 去 gather input），如果 `source.size_1 ≠ input.size_1`，分解出的 row/col 会映射到错误的物理位置，导致**静默数据错位**。

```python
# ❌ 危险：source 和 input 的 size_1 不同
def application(source, input, output):
    src_idx = source.offsets(0)           # 在 source 的 tile 内
    val = input[src_idx]                  # gather 用 source 的 size_1 分解 → 错误！

# ✅ 安全方案 1：所有 tensor 保持相同的 size_1
# （例如都 flatten 成 1D，block_size 统一 → size_1 = block_size）

# ✅ 安全方案 2：避免跨 tensor gather，改用 1D copy kernel
# kernel 只做 output = source，索引计算放在 wrapper 中
# 详见 Pattern 13 in CODE_TEMPLATES.md
```

**根本原因：** 九齿编译器为 `tensor[idx]` 生成的 Triton IR 假设所有 tensor 共享相同的 tile 几何（因为它们在同一 arrangement 中声明），但 gather 索引的分解需要基于**被索引 tensor** 的实际形状。

**规避策略：**
1. **同尺寸 tile**：确保 arrangement 中所有 tensor 的最后一维大小一致
2. **1D copy + wrapper scatter**：kernel 只做简单拷贝，复杂索引在 wrapper 中用 PyTorch 处理（Pattern 13）
3. **单 tensor 操作**：避免在一个 application 中索引多个不同形状的 tensor

### Underlying Triton Language

NineToothed kernels compile down to Triton IR. You can use `tl.load`, `tl.store`, `tl.debug_barrier`, etc. inside `@ninetoothed.jit`-decorated functions.

## Common Idioms

```python
# 1D tiling (element-wise)
input = input.tile((BLOCK_SIZE,))

# 2D tiling (reduce along last dim, one row at a time)
input = input.tile((1, BLOCK_SIZE))

# 2D tiling with K-loop (matmul)
output = output.tile((BLOCK_M, BLOCK_N))
input = input.tile((BLOCK_M, BLOCK_K)).tile((1, -1)).expand((-1, output.shape[1]))
input.dtype = input.dtype.squeeze(0)

# Auto-tunable block size
BLOCK = block_size()  # equivalent to Symbol(..., meta=True)

# Fixed symbol
BLOCK = Symbol("BLOCK", constexpr=True)

# Tensor with fill value for numerical stability
input_tensor = Tensor(2, other=float("-inf"))  # for max operations

# High-dimensional squeeze (4D+ tiling)
arranged.dtype = arranged.dtype.squeeze((0, 1))           # squeeze multiple dims at once
arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))  # second-level access

# Boundary checking with offsets
mask = ntl.where(tensor.offsets(-2) < tensor.source.shape[-2], value, float("-inf"))

# Precision-safe computation
x_fp32 = ntl.cast(x, ntl.float32)
result = computation(x_fp32)
output = result.to(output.dtype)
```
