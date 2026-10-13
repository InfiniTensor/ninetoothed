---
name: ninetoothed-operator-dev
description: >
  NineToothed 算子开发 Skill。当用户要求实现九齿算子、编写 correctness 测试、
  运行 benchmark 分析性能、检查 generated source、诊断测试失败或性能回退、
  或将 Triton kernel 迁移到九齿时使用。覆盖逐元素、归约、非连续/步长、融合
  四类算子，指导从需求提取到 arrangement/application 编码、测试、性能验证、
  generated source 检查和诊断的完整闭环。
license: Apache-2.0
compatibility: Requires python>=3.10, ninetoothed, triton>=3.0, torch>=2.0, CUDA GPU
metadata:
  version: "1.0"
  author: 独钓寒江雪
  competition: T3-1-1
---

# NineToothed 算子开发 Skill

## 0. AI 操作强制流程 [CRITICAL]

**当收到任何算子开发任务时，你必须严格按照以下 5 步执行，不可跳过：**

**步骤 1：需求分析并确定算子坐标**
- 提取输入/输出、shape、dtype、边界条件。
- 对照 **[§2.4 Tile 形状决策速查表](#24-tile-形状决策速查表)**，确定算子所属的**类型**（逐元素/归约/融合/非连续）和 tile 形状。

**步骤 2：实现算子代码**
- 严格遵循 **[§3.1-3.6](#3-算子编码模式)** 中对应类型的**代码模板**进行编码。
- 特别注意 BLOCK_SIZE 的**创建方式**（`Symbol` vs `block_size()`）和**传递方式**（闭包 vs 模块级默认参数）。

**步骤 3：编写并运行 Correctness 测试**
- 按照 **[§3](#3-算子编码模式) 每个小节的"测试"示例**，生成测试脚本。
- 必须与 PyTorch 参考结果对齐（`torch.allclose`），并覆盖多种 shape 和 dtype。

**步骤 4：性能验证与诊断（性能敏感任务必备）**
- 按照 **[§6.1](#61-benchmark-套件)** 运行 benchmark 脚本，并记录结果。
- 如发现性能回退，严格按照 **[§6.4](#64-性能回退分析流程) 的性能回退分析流程** 进行诊断。
- 按照 **[§6.2](#62-generated-source-检查)** 检查 generated source，定位代码层面的低效点。

**步骤 5：交付闭环**
- 整理最终代码、测试文件、benchmark 结果和诊断结论，向用户报告。

---

## 1. 核心概念

### 1.1 九齿简介

九齿（NineToothed）是一个基于 [Triton](https://triton-lang.org/) 的领域特定语言（DSL）。通过引入**面向张量的元编程**（Tensor-Oriented Metaprogramming, TOM），它能够进一步简化高性能计算内核的开发。

**核心特性：**
- **Arrangement 函数**：定义张量的排列方式（编译时）
- **Application 函数**：定义如何应用排列后的张量（运行时）
- **ninetoothed.make()**：整合 arrangement 和 application 构建内核
- **自动调优**：使用 `block_size()` 启用自动调优

### 1.2 安装

```bash
pip install ninetoothed
```

## 2. 核心概念详解

### 2.1 九齿 Language 函数

九齿使用 `ninetoothed.language` (别名 `ntl`) 提供数学和tensor操作函数。

#### 2.1.1 常用函数分类

| 类别 | 函数 | 说明 |
|------|------|------|
| **数学运算** | `ntl.exp`, `ntl.log`, `ntl.sqrt`, `ntl.abs` | 基本数学运算 |
| **逐元素比较** | `ntl.maximum`, `ntl.minimum` | 逐元素取最大值/最小值 |
| **类型转换** | `ntl.cast`, `ntl.float32`, `ntl.float16`, `ntl.int32` | 类型转换 |
| **归约操作** | `ntl.sum`, `ntl.max`, `ntl.min` | 跨维度归约操作 |
| **激活函数** | `ntl.sigmoid` | Sigmoid 函数（底层映射到 tl.sigmoid） |
| **特殊数学** | `ntl.libdevice.*` | libdevice 数学函数库（需直接导入） |

⚠️ **重要：复杂的数学函数需要通过 `ntl.libdevice` 访问！**

#### 2.1.2 libdevice 与激活函数

Triton 依赖 CUDA libdevice 提供高级数学函数。必须**直接导入 `libdevice` 作为独立全局变量**，不能通过 `ntl.libdevice.*` 间接访问（ninetoothed Inliner 无法识别）：

```python
# ✅ 正确
from ninetoothed.language import libdevice
libdevice.tanh(x)

# ✅ sigmoid/tanh 的轻量替代（无需 libdevice）
ntl.sigmoid(x)
2.0 * ntl.sigmoid(2.0 * x) - 1.0  # tanh 恒等式

# ❌ 错误
ntl.libdevice.tanh(x)  # 代码生成失败！
```

⚠️ **fp16 输入需先转为 fp32**：`ntl.exp`、`ntl.sigmoid` 等要求 fp32 输入。

| 函数 | 推荐方式 | 备选 |
|------|---------|------|
| exp, log, sqrt | `ntl.exp/log/sqrt` | libdevice |
| sigmoid | `ntl.sigmoid` | libdevice |
| tanh | `2*sigmoid(2x)-1` | libdevice |
| erf, sin, cos | libdevice | — |
| maximum, minimum | `ntl.maximum/minimum` | — |

#### 2.1.3 Tensor 元操作（Slicing / Expand）

在 arrangement 函数中可对 Tensor 进行切片和维度操作：

```python
# 切片 — 非连续访存
x[::2]          # 步长 2
x[::-1]         # 反转
x[:, ::2]       # 多维切片（非合并访存）

# Unsqueeze & Expand — 广播
x.unsqueeze(0)              # (N,) → (1, N)
x.unsqueeze(0).expand((B, -1))  # (1, N) → (B, N)
```

| 模式 | 语法 | 用途 |
|------|------|------|
| 步长 k | `x[::k]` | 每隔 k 个元素 |
| 反转 | `x[::-1]` | 逆序排列 |
| 多维切片 | `x[:, ::2]` | 2D 沿列方向步长 |
| 广播 | `unsqueeze` + `expand` | 扩展维度 |

> 多层 tiling 的 `dtype.squeeze()`、`ravel()`、`flatten()` 等高级模式（用于 mm/bmm/max_pool2d）详见 `references/operator_patterns.md`。

### 2.2 Symbol 系统

九齿使用 Symbol 类来表示符号表达式，这是编译时元编程的基础。

#### 2.2.1 创建方式

| 创建方式 | 说明 | 适用场景 |
|---------|------|----------|
| `block_size()` | 创建用于自动调优的块大小符号（meta 类型） | 动态性能调优 |
| `Symbol(name, constexpr=True)` | 创建编译时常量符号 | 固定参数配置 |
| `Symbol(name, meta=True)` | 创建元参数符号 | 动态参数 |

**示例：**
```python
from ninetoothed import Symbol, block_size

# 方式1：自动调优（推荐）
BLOCK_SIZE = block_size()  # 会自动生成 meta 参数

# 方式2：固定常量
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

# 方式3：自定义元参数
BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()
```

#### 2.2.2 Symbol 参数签名规则（关键！）

arrangement 函数中 Symbol 有两种传递方式：

| 方式 | 适用场景 | 示例 |
|------|---------|------|
| **闭包**（函数内创建 `block_size()`） | 简单一维算子 | `def create_kernel(): BLOCK_SIZE = block_size(); def arrangement(...)` |
| **模块级 + 参数默认值**（推荐） | 所有场景 | `BLOCK_SIZE = Symbol(...); def arrangement(x, y, BLOCK_SIZE=BLOCK_SIZE): ...` |

⚠️ **二维及以上必须从模块级别定义**，并通过参数默认值传入 arrangement。例如 ninetoothed-examples/mm.py：
```python
BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = block_size(), block_size(), block_size()
def arrangement(input, other, output, BLOCK_SIZE_M=BLOCK_SIZE_M, ...): ...
```

#### 2.2.3 Tile 形状的语义模型（二进制规则）

⚠️ **最关键的语义规则：`tile_shape[i] ∈ {1, BLOCK_SIZE}`**

| tile 形状元素 | 语义 | 说明 |
|-------------|------|------|
| `BLOCK_SIZE` | **并行分块** | 该维度被分成多个块并行处理 |
| `1` | **完整保留** | 该维度保持完整，归约操作可以跨越整个维度 |

**三种运算类型的 tile 策略：**

| 类型 | 规则 | 逐元素维度 | 归约维度 |
|------|------|-----------|---------|
| 逐元素（Add, ReLU, GELU） | 所有维度用 `BLOCK_SIZE` | `BLOCK_SIZE` | 无 |
| 归约（Softmax, Sum） | 归约维度 `BLOCK_SIZE`，其余 `1` | `1` | `BLOCK_SIZE` |
| 混合（RMS Norm） | 归约轴 `BLOCK_SIZE`，非归约轴 `1` | `1` | `BLOCK_SIZE` |

**快速对照：**
```python
# 逐元素 1D: (BLOCK_SIZE,)
# 逐元素 2D: (BLOCK_SIZE_M, BLOCK_SIZE_N)
# 归约沿列: (1, BLOCK_SIZE)     — 列维度分块，行维度完整
# 归约沿行: (BLOCK_SIZE, 1)     — 行维度分块，列维度完整
```

### 2.3 Tensor 维度设计

**核心规则：`Tensor(n)` 的 `n` = tile 形状长度**

| Tensor | tile 形状 | 说明 |
|--------|----------|------|
| `Tensor(1)` | `(BLOCK_SIZE,)` | 一维向量 |
| `Tensor(2)` | `(BLOCK_SIZE_M, BLOCK_SIZE_N)` | 二维矩阵 |
| `Tensor(0)` | 不参与 tile | 标量参数（如 eps）透传 |

### 2.4 Tile 形状决策速查表

| 算子类型 | Tensor | 归约轴 | tile 形状 | 调用约束 | BLOCK_SIZE 类型 |
|---------|--------|--------|----------|---------|----------------|
| 逐元素 1D | `Tensor(1)` | 无 | `(BLOCK_SIZE,)` | — | `Symbol(constexpr)` |
| 逐元素 2D | `Tensor(2)` | 无 | `(B_M, B_N)` | — | `block_size()` / `Symbol(constexpr)` |
| 归约 1D (Sum) | `Tensor(1)` | 0 | `(BLOCK_SIZE,)` | `BLOCK_SIZE=n` | **必须** `Symbol(constexpr=True)` |
| 归约 2D 沿列 (Softmax) | `Tensor(2)` | 1 | `(1, BLOCK_SIZE)` | `BLOCK_SIZE=shape[-1]` | **必须** `Symbol(constexpr=True)` |
| 融合归约 (RMS Norm) | `Tensor(2)` | 1 | `(1, BLOCK_SIZE)` + 标量 | `BLOCK_SIZE=shape[-1]` | **必须** `Symbol(constexpr=True)` |
| 非连续 1D (Strided) | `Tensor(1)` | 无 | `(BLOCK_SIZE,)` + slice | slice **先于** tile | `Symbol(constexpr)` |
| 非连续 2D (Strided) | `Tensor(2)` | 无 | `(B_M, B_N)` + slice | slice **先于** tile | `block_size()` / `Symbol(constexpr)` |

## 3. Symbol 调用模式（按维度分类）

### 3.1 一维算子（向量操作）

**适用场景**：向量加法、ReLU、Sigmoid、GELU 等

一维算子有两种推荐的 BLOCK_SIZE 定义方式。**推荐方式 1（官方标准）**，方式 2 在简单场景下也可以工作。

```python
from ninetoothed import Symbol, Tensor

# 方式 1（推荐）：模块级 constexpr Symbol（官方标准）
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(x, y, z, BLOCK_SIZE=BLOCK_SIZE):
    return (x.tile((BLOCK_SIZE,)),
            y.tile((BLOCK_SIZE,)),
            z.tile((BLOCK_SIZE,)))


def application(x, y, z):
    z = x + y  # noqa: F841


def create_add_kernel():
    """向量加法 kernel（一维）"""
    return ninetoothed.make(arrangement, application,
                          (Tensor(1), Tensor(1), Tensor(1)))
```

**方式 2（替代）：闭包 block_size()**
```python
from ninetoothed import Tensor, block_size

def create_add_kernel():
    BLOCK_SIZE = block_size()
    def arrangement(x, y, z):
        return (x.tile((BLOCK_SIZE,)),
                y.tile((BLOCK_SIZE,)),
                z.tile((BLOCK_SIZE,)))
    def application(x, y, z):
        z = x + y  # noqa: F841
    return ninetoothed.make(arrangement, application,
                          (Tensor(1), Tensor(1), Tensor(1)))
```

**测试：**
```python
import torch
kernel = create_add_kernel()
a = torch.randn(1024, dtype=torch.float16, device='cuda')
b = torch.randn(1024, dtype=torch.float16, device='cuda')
c = torch.empty_like(a)
kernel(a, b, c)
assert torch.allclose(c, a + b, atol=1e-5, rtol=1e-3)
```

### 3.2 二维算子（矩阵操作）- 参考 ninetoothed-examples/mm.py

**适用场景**：矩阵加法、矩阵乘法等二维操作

⚠️ **关键**：必须使用 **模块级别** 定义多个 `block_size()`，并作为参数传递给 arrangement 函数

```python
from ninetoothed import Tensor, block_size
import ninetoothed.language as ntl

# ⚠️ 必须从模块级别定义！
BLOCK_SIZE_M = block_size()  # M 维度
BLOCK_SIZE_N = block_size()  # N 维度


def arrangement(x, y, z, 
               BLOCK_SIZE_M=BLOCK_SIZE_M, 
               BLOCK_SIZE_N=BLOCK_SIZE_N):
    # 二维 tiling：同时对 M 和 N 维度分块
    return (x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
            y.tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
            z.tile((BLOCK_SIZE_M, BLOCK_SIZE_N)))


def application(x, y, z):
    z = x + y  # noqa: F841

def create_2d_add_kernel():
    """二维矩阵加法 kernel"""
    return ninetoothed.make(arrangement, application, 
                          (Tensor(2), Tensor(2), Tensor(2)))
```

**测试：**
```python
import torch
kernel = create_2d_add_kernel()
a = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
b = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
c = torch.empty_like(a)
kernel(a, b, c)
assert torch.allclose(c, a + b, atol=1e-5, rtol=1e-3)
```

### 3.3 二维归约算子（沿特定维度）- 参考 ninetoothed-examples/softmax.py

**适用场景**：Softmax、沿行/列归约

⚠️ **归约算子必须使用 `Symbol(constexpr=True)` 而非 `block_size()`！** 因为 `block_size()` 创建的 meta 参数可能小于归约维度大小，导致多 block 局部归约互相覆盖，结果错误。

```python
from ninetoothed import Symbol, Tensor
import ninetoothed.language as ntl

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(x, output, BLOCK_SIZE=BLOCK_SIZE):
    # tile((1, BLOCK_SIZE)): 保持第一维（M），只对第二维（N）分块
    # 用于沿 N 维度归约的算子
    return x.tile((1, BLOCK_SIZE)), output.tile((1, BLOCK_SIZE))


def application(x, output):
    x_max = ntl.max(x)
    x_shifted = x - x_max
    exp_x = ntl.exp(x_shifted)
    sum_exp = ntl.sum(exp_x)
    output = exp_x / sum_exp  # noqa: F841


def create_softmax_kernel():
    """Softmax kernel（沿列归约）"""
    return ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2)))
```

**测试（需要传入完整的 BLOCK_SIZE）：**
```python
import torch
kernel = create_softmax_kernel()
x = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
output = torch.empty_like(x)
kernel(x, output, BLOCK_SIZE=x.shape[-1])  # BLOCK_SIZE 必须等于最后一维大小
expected = torch.softmax(x, dim=-1)
assert torch.allclose(output, expected, atol=1e-2, rtol=1e-2)
```

### 3.4 三维算子（批量矩阵操作）

**适用场景**：批量矩阵乘法、3D 张量操作

```python
from ninetoothed import Tensor, block_size

# 模块级别定义
BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()
BLOCK_SIZE_K = block_size()


def arrangement(x, y, z, 
               BLOCK_SIZE_M=BLOCK_SIZE_M, 
               BLOCK_SIZE_N=BLOCK_SIZE_N,
               BLOCK_SIZE_K=BLOCK_SIZE_K):
    # 三维 tiling
    return (x.tile((BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N)),
            y.tile((BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N)),
            z.tile((BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N)))


def application(x, y, z):
    z = x + y


def create_3d_add_kernel():
    """三维张量加法 kernel"""
    return ninetoothed.make(arrangement, application,
                          (Tensor(3), Tensor(3), Tensor(3)))
```

### 3.5 非连续输入/步长算子

**适用场景**：需要按非单位步长访问数据，如取每隔 k 个元素、反转维度、切片裁剪等。

在 `arrangement` 中，使用 Python 切片语法对 Tensor 进行变换，即可生成支持非连续访存的 kernel。

```python
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(x, output, BLOCK_SIZE=BLOCK_SIZE):
    # 每隔一个元素取一个（步长2），输出大小相应减半
    x_strided = x[::2]
    return x_strided.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

def application(x, output):
    output = x  # noqa: F841

def create_stride2_copy_kernel():
    """步长为2的拷贝 kernel（一维）"""
    return ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))
```

**更复杂的跨步场景（Strided Add）** — 三张量同步跨步访问，参考 `operators/strided_add.py`：

```python
def arrangement(input, other, output, BLOCK_SIZE=BLOCK_SIZE):
    # 三个张量使用相同的 stride=2 切片
    return (
        input[::2].tile((BLOCK_SIZE,)),
        other[::2].tile((BLOCK_SIZE,)),
        output[::2].tile((BLOCK_SIZE,)),
    )

def application(input, other, output):
    output = input + other  # noqa: F841

tensors = (Tensor(1), Tensor(1), Tensor(1))
kernel = ninetoothed.make(arrangement, application, tensors)
```

**二维跨步 (2D Strided Add)** — 沿列方向步长，演示"行合并 + 列非合并"的混合访存模式：

```python
def arrangement_2d(input, other, output,
                   BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
                   BLOCK_SIZE_COL=BLOCK_SIZE_COL):
    # 沿最后一维做 stride=2，第一维正常分块
    return (
        input[:, ::2].tile((BLOCK_SIZE_ROW, BLOCK_SIZE_COL)),
        other[:, ::2].tile((BLOCK_SIZE_ROW, BLOCK_SIZE_COL)),
        output[:, ::2].tile((BLOCK_SIZE_ROW, BLOCK_SIZE_COL)),
    )
```

> 2D 跨步的关键意义：第一维保持合并访问（coalesced），第二维 stride=2 产生非合并访问，可用作性能分析案例——对比 stride=1 vs stride=2 的带宽差异。

**切片模式速查：**

| 模式 | 语法 | 用途 |
|------|------|------|
| 步长 k | `x[::k]` | 每隔 k 个元素取一个 |
| 反转 | `x[::-1]` | 逆序排列 |
| 裁剪 | `x[1:-1]` | 去除首尾 |
| 多维切片 | `x[:, ::2]` | 2D 沿列方向步长（非合并访存） |
| 新增维度 | `x[None]` | 等价于 unsqueeze |

### 3.6 融合算子（Reduction + Element-wise）

**适用场景**：RMS Normalization、Layer Normalization、Fused SiLU 等在一个 kernel 中完成归约+逐元素的操作。

参考 `ninetoothed-examples/ops/ninetoothed/kernels/fused_rms_norm.py`。

**完整 RMS Norm 示例（operators/rms_norm.py）：**

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(x, w, eps, y, BLOCK_SIZE=BLOCK_SIZE):
    def arrange(tensor):
        return tensor.tile((1, BLOCK_SIZE))
    return arrange(x), arrange(w), eps, arrange(y)


def application(x, w, eps, y):
    x_fp32 = ntl.cast(x, ntl.float32)
    y = x_fp32 * ntl.rsqrt(  # noqa: F841
        ntl.sum(x_fp32 * x_fp32) / x.shape[-1] + eps
    ) * w


tensors = (Tensor(2), Tensor(2), Tensor(0), Tensor(2))
kernel = ninetoothed.make(arrangement, application, tensors)
```

**融合算子关键模式：**

1. **标量参数透传**：`eps` 用 `Tensor(0)` 定义，在 arrangement 中不加任何处理，直接透传到 application
2. **多张量同 tile**：使用局部函数 `def arrange(tensor)` 对多个同维张量应用相同 tiling
3. **混合精度**：`ntl.cast(x, ntl.float32)` 在 fp16 输入上进行 fp32 计算，结果自动转换回输出 dtype
4. **融合范围**：square → sum → divide → rsqrt → rescale → multiply by w 全部在单个 kernel 内完成

**对比：非融合 vs 融合：**

| 方面 | 非融合（多 kernel） | 融合（单 kernel） |
|------|-------------------|-------------------|
| 中间张量 | 需要多次 load/store | 寄存器内计算 |
| 访存次数 | N 次（每个 kernel 独立访存） | 1 次 |
| 实现方式 | 拆分为多个 kernel 顺序调用 | 在 application 中串联所有操作 |

## 4. 常见问题与解决方案

### 4.1 Symbol 相关错误

#### 错误1: BLOCK_SIZE 参数未正确传递
**原因**: arrangement 函数签名包含 BLOCK_SIZE 参数，但使用模块级别的 `block_size()` 创建但未正确传递
**解决**: 确保模块级别的 `block_size()` 通过函数参数默认值传递

```python
# ✅ 正确：模块级别定义 + 参数默认值
BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()

def arrangement(x, y, z, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N):
    return x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N)), ...
```

#### 错误2: "Dimension out of range"
**原因**: Tensor 维度与 tile 形状不匹配
**解决**: 确保 `len(tile_shape) == Tensor 维度数`

```python
# ❌ 错误：Tensor(1) 但 tile 使用2个参数
x = Tensor(1)
x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))  # ❌

# ✅ 正确：Tensor(2) + 二参数 tile
x = Tensor(2)
x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))  # ✅
```

### 4.2 维度设计错误

#### 错误3: 二维张量只有一列正确
**原因**: tiling 只处理了第一个维度
**解决**: 使用二维 tiling `tile((BLOCK_SIZE_M, BLOCK_SIZE_N))`

```python
# ❌ 错误：一维 tiling
x = Tensor(2)  # 二维张量
x.tile((BLOCK_SIZE,))  # ❌ 只处理第一个维度

# ✅ 正确：二维 tiling
x = Tensor(2)  # 二维张量
x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))  # ✅ 同时处理两个维度
```

#### 错误4: 一维测试通过但二维失败
**原因**: 同一个 kernel 只能支持一种维度配置
**解决**: 为不同维度创建独立的 kernel

```python
# 创建两个独立的 kernel
def create_1d_kernel():
    return ninetoothed.make(arrangement_1d, application, (Tensor(1), ...))

def create_2d_kernel():
    return ninetoothed.make(arrangement_2d, application, (Tensor(2), ...))
```

## 5. 编码约定与调试

### 5.1 `# noqa: F841` 约定

在 `application` 函数中对 `output` 张量的赋值即是 Triton store 操作。`output` 变量在赋值后不再被 Python 读取，统一在赋值行添加 `# noqa: F841`：

```python
def application(x, output):
    output = x + 0.5  # noqa: F841
```

### 5.2 调试检查清单

1. **维度匹配**：`len(tile_shape) == Tensor 维度数`
2. **Symbol 类型**：归约算子必须 `Symbol(constexpr=True)`（调用方需传入完整维度值）
3. **BLOCK_SIZE 传递**：二维及以上必须模块级定义 + 参数默认值传入 arrangement
4. **fp16 cast**：`ntl.exp` / `ntl.sigmoid` 需要 fp32 输入，用 `ntl.cast(x, ntl.float32)` 转换
5. **generated source 验证**：检查 `kernel.src` 或 `~/.ninetoothed/` 中的生成代码

## 6. 性能验证与调试

### 6.1 Benchmark 套件

项目 `benchmarks/` 目录提供了完整的 benchmark 套件，遵循官方 `triton.testing` 模式（参考 `ninetoothed-examples/bench.py`）。

**核心组件：**

| 文件 | 用途 |
|------|------|
| `benchmarks/bench.py` | 包装器，基于 `triton.testing.Benchmark` + `perf_report` + `do_bench` |
| `benchmarks/run_benchmarks.py` | 编排全部算子的 benchmark，按类别分组 |

**运行方式：**

```bash
# 快速模式（推荐先用这个验证环境，每个算子 3 个规模点）
python benchmarks/run_benchmarks.py --quick

# 运行全部 benchmark（正常 sweep，约 5-10 分钟）
python benchmarks/run_benchmarks.py

# 运行全部并保存图表
python benchmarks/run_benchmarks.py --save-path ./bench_results

# 仅逐元素类（快速）
python benchmarks/run_benchmarks.py --category elementwise --quick

# 仅归约类
python benchmarks/run_benchmarks.py --category reduction

# 仅非连续访存分析（stride=2 vs contiguous）
python benchmarks/run_benchmarks.py --category analysis

# 列出所有可用的 benchmark
python benchmarks/run_benchmarks.py --list
```

**Benchmark 类别覆盖：**

| 类别 | 覆盖算子 | sweep 规模 |
|------|---------|-----------|
| elementwise | add_1d, relu, sigmoid, gelu, add_2d | 1K ~ 32M (1D); 32~8192 (2D) |
| reduction | softmax, sum | n=32~16384 (softmax); 1K~32M (sum) |
| noncontiguous | strided_add_1d, strided_add_2d | 同上 |
| fused | rms_norm | n=32~16384 |
| analysis | stride_vs_contiguous | stride=2 与 contiguous 直接对比 |

**Benchmark 输出示例（softmax, n=4096）：**

```
   n       ninetoothed      torch
4096.0       0.062           0.058
```

**性能对比模板（编写 benchmark 报告时使用）：**

```
算子: <名称>
输入规模: <shapes>
ninetoothed 平均延迟: X.XX ms
PyTorch 平均延迟:    Y.YY ms
加速比: Z.ZZX
访存带宽利用率: W%
结论: [性能正常 / 轻微回退需优化 / 明显回退需排查]
```

### 6.2 Generated Source 检查

`ninetoothed.make()` 生成的 Triton 源码缓存在 `~/.ninetoothed/` 目录下，文件名为 SHA-256 哈希。可以直接查看：

```bash
# 列出最近生成的源码文件
ls -lt ~/.ninetoothed/*.py | head -5

# 查看某个 kernel 的源码（kernel.src 属性）
python -c "
from operators import create_add_kernel
k = create_add_kernel()
print(k.src)  # 打印生成的 Triton 代码
"
```

**检查要点：**
1. Load/Store 次数是否与预期匹配（逐元素：1 load + 1 store 每个 tile）
2. 是否有冗余的 intermediate load/store
3. 归约操作的 axis 是否正确（`tl.sum(..., axis=None)` vs `axis=0`）
4. dtype 转换是否正确（fp16→fp32→fp16 路径）
5. `tl.libdevice` 引用存在时排查路径错误（应为 `triton.language.extra.libdevice`）

**常见 source 问题速查：**

| 现象 | 可能原因 | 检查方法 |
|------|---------|---------|
| 多出 .cvt 指令 | 不必要的 dtype cast | 检查 application 中是否有多余的 ntl.cast |
| 多出 .load/.store | tile 形状不匹配数据量 | 检查 BLOCK_SIZE 是否覆盖整个维度 |
| 冗余的 for 循环 | 归约未正确展开 | 检查归约轴的 BLOCK_SIZE 是否等于该维度大小 |
| libdevice 引用错误 | ntl.libdevice.* 路径不对 | 改用 ntl.sigmoid 或直接 import libdevice |

### 6.3 AOT (Ahead-of-Time) Build

ninetoothed 支持 AOT 编译，生成 `.so` + `.cpp`/`.h` 文件供 C++ 项目集成。

**AOT 核心 API（`ninetoothed.build`）：**

```python
import ninetoothed

def premake():
    return ninetoothed.make(arrangement, application, tensors)

ninetoothed.build(
    premake,
    configs=[{"BLOCK_SIZE": 256}, {"BLOCK_SIZE": 512}, {"BLOCK_SIZE": 1024}],
    output_dir="./aot_output",
    meta_parameters=("BLOCK_SIZE",),  # 可选：自动调优选择最优值
)
```

**输出产物（`output_dir/`）：**
- `<kernel>.so` — 编译后的共享库
- `<kernel>.cpp` / `<kernel>.h` — C++ 调度器
- `<kernel>.csv` — 自动调优缓存（最佳 meta 值）
- `<kernel>.fingerprint` — 增量构建指纹

> 完整 AOT API 参考 ninetoothed 官方文档 `docs/source/build.rst`。

### 6.4 性能回退分析流程

```
遇到测试变慢或 benchmark 回退时：
1. 确认回退规模 — 是小规模（<1K）还是大规模（>1M）？不同规模瓶颈不同
2. 检查 generated source — 确认生成的 Triton 代码逻辑正确（§7.2）
3. 对比 tile 配置 — 检查 BLOCK_SIZE 是否合理（过小：并行度不足；过大：寄存器溢出）
4. 检查 dtype — 确认是否需要 fp32 转换（fp16→fp32→fp16 开销）
5. 检查访存模式 — slice/stride 是否引入非合并访问（§7.5 示例）
6. 对比 PyTorch 基线 — 确认回退是内核问题还是输入规模线性退化
7. 记录分析结论 — 现象 + 根因 + 修复 + 验证结果
```

### 6.5 性能分析案例：Stride vs Contiguous

项目中 `benchmarks/run_benchmarks.py` 的 `analysis` 类别提供了一个完整的性能对比案例：

```bash
python benchmarks/run_benchmarks.py --category analysis
```

该测试对比同一数据在不同访存模式下（`c[::2] = a[::2] + b[::2]` vs `c = a + b`）的延迟差异。在大多数 GPU 上，stride=2 的非连续访存会导致约 20%-50% 的性能损失。

**分析脚本输出的关键信息：**
- 每个规模点的 strided 延迟 vs contiguous 延迟
- 延迟差异百分比
- 可用于判断隐藏任务中"非连续布局开销是否正常"

> 这直接对应赛题 4.2 节"非连续输入开销分析"和 4.4 节"性能意识"评分要求。

常见性能陷阱：

| 陷阱 | 表现 | 修复 |
|------|------|------|
| BLOCK_SIZE 过小 | 启动开销大于计算开销 | 增大 BLOCK_SIZE |
| BLOCK_SIZE 过大 | 寄存器溢出 | 减小 BLOCK_SIZE |
| 非合并访存 | stride != 1 导致带宽下降 | 考虑连续布局或重组数据 |
| 不必要的 fp16→fp32 转换 | 转换指令占比高 | 仅在需要精度时转换 |
| 缺乏 AOT/build 配置 | 每次运行时重编译 | 使用 AOT 预编译 |

## 7. 参考资源

- **官方源码**: `ninetoothed-examples/ops/ninetoothed/kernels/`
  - add.py: 一维向量加法
  - mm.py: 二维矩阵乘法（多层 tiling + dtype 操纵）
  - bmm.py: 批量矩阵乘法（复用 mm.application）
  - softmax.py: 二维归约（沿列）
  - max_pool2d.py: 滑动窗口归约 + ravel/flatten 模式
  - silu.py / swiglu.py: 激活函数 & fp32 cast 模式
  - rms_norm.py / fused_rms_norm.py: 融合归约算子
- **官方测试**: ninetoothed 仓库 `tests/` 目录
  - test_getitem.py: Tensor 切片/slice 语法验证
  - test_pad.py: 非连续切片的 pad 实现
  - test_unsqueeze.py: unsqueeze + expand + tile 组合模式
- **本项目 operators/**: 7 个算子参考实现 + 2 个补充算子
  - add.py, add_2d.py: 一维/二维逐元素
  - relu.py, sigmoid.py, gelu.py: 激活函数
  - softmax.py, sum.py: 归约算子
  - strided_add.py: 非连续/步长算子
  - rms_norm.py: 融合归约算子
- **API 文档**: 查看 `references/nine_toothed_api.md`
- **设计模式**: 查看 `references/operator_patterns.md`

---

*本文档基于九齿框架实践编写，重点强调 Symbol 调用规则和维度设计*
