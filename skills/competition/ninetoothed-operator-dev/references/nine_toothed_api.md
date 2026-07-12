# 九齿 API 参考

本文档记录已在实际算子中验证可用的九齿 API。所有内容均来自 SKILL.md 和 `operators/` 目录下的工作代码。

## 1. 安装与导入

### 1.1 安装

```bash
pip install ninetoothed
```

### 1.2 核心导入

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, Symbol, block_size
```

> ⚠️ 你**不能**使用 `nt.*` 命名空间。`ninetoothed` 模块只导出 `make()`, `build()`, `Tensor`, `Symbol`, `block_size`。
> 数学和归约原语全部在 `ninetoothed.language`（别名 `ntl`）中。

## 2. 核心类

### 2.1 Tensor 类

用于声明符号张量的**维度数**。不存储实际数据，仅描述 shape/stride。

```python
from ninetoothed import Tensor

# 声明一维张量
x = Tensor(1)

# 声明二维张量
x = Tensor(2)

# 声明标量（不参与 tile，如 eps）
x = Tensor(0)
```

**核心规则**：`Tensor(n)` 的 `n` = tile 形状的长度。例如 `Tensor(2)` 必须配 `tile((B_M, B_N))`。

**属性**：
- `shape`：张量的形状（符号表达式）

### 2.2 Symbol 类

用于创建编译时符号变量。

```python
from ninetoothed import Symbol

# 编译时常量（固定值，调用时传入）
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

# 元参数（用于自动调优场景，需要配合具体框架使用）
BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)
```

**关键规则**：
- **归约算子必须用 `Symbol(constexpr=True)`**，调用时传入完整维度值
- **模块级定义 + 参数默认值传入 arrangement**（二维及以上强制如此）

### 2.3 block_size 函数

创建用于自动调优的块大小符号。

```python
from ninetoothed import block_size

# 签名: block_size(lower_bound=None, upper_bound=None)
BLOCK_SIZE = block_size()
BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()
```

适用于逐元素算子和简单分块场景。不适用于归约算子。

## 3. 元操作

元操作是对符号张量执行的编译时操作，写在 `arrangement` 函数中。

### 3.1 tile 操作

将张量分块。这是最核心的元操作。

```python
x.tile((BLOCK_SIZE,))                    # 1D 分块
x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))     # 2D 分块
x.tile((1, BLOCK_SIZE))                  # 保留第一维，只对第二维分块
```

### 3.2 切片

在 tile 之前做切片，实现非连续访存。

| 模式 | 语法 | 用途 |
|------|------|------|
| 步长 k | `x[::k]` | 每隔 k 个元素 |
| 反转 | `x[::-1]` | 逆序排列 |
| 裁剪 | `x[1:-1]` | 去除首尾 |
| 多维切片 | `x[:, ::2]` | 2D 沿列方向步长 |
| 新增维度 | `x[None]` | 等价于 unsqueeze |

**关键规则**：切片**先于** tile。即 `x[::2].tile((BLOCK_SIZE,))`。

### 3.3 expand 操作

扩展张量维度，用于广播场景。

```python
x.unsqueeze(0).expand((B, -1))  # (N,) → (1, N) → (B, N)
```

### 3.4 squeeze 操作

压缩维度。

```python
x.squeeze(0)  # (1, K) → (K,)
```

### 3.5 permute 操作

交换维度。

```python
x.permute((2, 0, 1))  # (A, B, C) → (C, A, B)
```

> 多层 tiling 的高级模式（ravel, flatten 等）用于 mm/bmm/max_pool2d 等复杂算子，详见 `references/operator_patterns.md`。

## 4. 运行时语言（ntl.*）

在 `application` 函数中使用。`ntl.*` 函数是 ninetoothed 代码生成器在编译时解析的符号名——它们在 Python 运行时不一定以普通函数形式存在（`hasattr(ntl, 'sum')` 可能返回 `False`），但在 `make()` 编译时被正确替换为 Triton 原语。

> ⚠️ 本章标注「已验证」指该 API 在 `operators/` 目录的至少一个算子中实际使用并通过 correctness 测试。标注「未验证」指 SKILL.md 提及但无算子使用——可能存在但未确认。

### 4.1 数学运算（已验证 ✅）

在至少一个算子中实际使用的 API：

```python
c = ntl.exp(a)       # 指数（需 fp32 输入）— gelu.py, softmax.py
c = ntl.sigmoid(a)   # Sigmoid（需 fp32 输入）— sigmoid.py, gelu.py
c = ntl.rsqrt(a)     # 平方根倒数 — rms_norm.py
c = ntl.maximum(a, b)  # 逐元素取最大值 — relu.py
```

基本运算符（`+`, `-`, `*`, `/`）在所有算子中使用。

未在算子中使用、但 SKILL.md 列出的 API（⚠️ 未验证）：

```
ntl.sqrt(a)      # 平方根 — 未在任何算子中使用
ntl.abs(a)       # 绝对值 — 未在任何算子中使用
ntl.minimum(a, b)  # 逐元素取最小值 — 未在任何算子中使用
```

### 4.2 归约操作（已验证 ✅）

```python
c = ntl.sum(a)    # 对 tile 内所有元素求和 — softmax.py, sum.py, rms_norm.py
c = ntl.max(a)    # 对 tile 内所有元素取最大值 — softmax.py
```

未使用但对称存在（⚠️ 未验证）：
```
ntl.min(a)    # 对 tile 内所有元素取最小值 — 未在任何算子中使用
```

> ⚠️ `ntl.sum` / `ntl.max` **不接受 dim 参数**，总是对 tile 内所有元素归约。

### 4.3 类型转换（已验证 ✅）

```python
x_fp32 = ntl.cast(x, ntl.float32)   # 转为 fp32 — gelu.py, sigmoid.py, rms_norm.py
```

未使用但对称存在（⚠️ 未验证）：
```python
x_fp16 = ntl.cast(x, ntl.float16)   # 未在算子中使用
x_i32  = ntl.cast(x, ntl.int32)     # 未在算子中使用
```

> ⚠️ 数据类型（`float32`, `float16`, `int32`）在 `ninetoothed` 模块中定义（`ninetoothed.float32`），但通过 `ntl.float32` 访问也能被代码生成器解析。

### 4.4 条件分支与创建操作（⚠️ 未验证）

以下 API 在 SKILL.md 中提及但**未在任何算子中使用**，理论上可通过代码生成器工作但未确认：

```python
output = ntl.where(condition, true_value, false_value)
zeros = ntl.zeros(shape, dtype=ntl.float32)
c = ntl.dot(a, b)     # 矩阵乘法（官方示例 mm.py 中使用）
```

> 若评测任务中 AI agent 需要使用这些 API，建议先在测试环境验证能否正常工作。

### 4.7 libdevice 函数

对于九齿未直接提供的数学函数（tanh, erf, sin, cos 等），需通过 CUDA libdevice：

```python
# ✅ 直接导入为独立全局变量
from ninetoothed.language import libdevice
libdevice.tanh(x)

# ❌ 不能通过 ntl.libdevice.* 间接访问（代码生成失败）
```

也可用恒等式替代（推荐，避免 libdevice 依赖）：

| 目标函数 | 替代恒等式 |
|---------|-----------|
| tanh(x) | `2.0 * ntl.sigmoid(2.0 * x) - 1.0` |

> 完整 libdevice 注意事项见 SKILL.md §2.1.2。

## 5. 内核构建

### 5.1 ninetoothed.make 函数

整合 arrangement 和 application 构建内核。**唯一的内核构建方式**。

```python
import ninetoothed
from ninetoothed import Tensor, Symbol

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(x, output, BLOCK_SIZE=BLOCK_SIZE):
    return x.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

def application(x, output):
    output = x + 0.5  # noqa: F841

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))
```

### 5.2 调用内核

```python
import torch

x = torch.randn(1024, dtype=torch.float16, device='cuda')
output = torch.empty_like(x)

kernel(x, output)  # 或 kernel(x, output, BLOCK_SIZE=1024)
```

## 6. AOT 编译

ninetoothed 支持 Ahead-of-Time 编译，生成 `.so` + `.cpp`/`.h` 文件：

```python
import ninetoothed

def premake():
    return ninetoothed.make(arrangement, application, tensors)

ninetoothed.build(
    premake,
    configs=[{"BLOCK_SIZE": 256}, {"BLOCK_SIZE": 512}, {"BLOCK_SIZE": 1024}],
    output_dir="./aot_output",
    meta_parameters=("BLOCK_SIZE",),
)
```

输出产物在 `output_dir/` 中：`.so`（共享库）、`.cpp`/`.h`（C++ 调度器）、`.csv`（自动调优缓存）。

> 完整 AOT API 参考 ninetoothed 官方文档。

## 7. 编码约定

### 7.1 `# noqa: F841` 约定

`application` 中对输出张量的赋值是 Triton store 操作。输出变量赋值后不再被 Python 读取，需加注释：

```python
def application(x, output):
    output = x + 0.5  # noqa: F841
```

### 7.2 Symbol 传递规则

| 场景 | 方式 | 示例 |
|------|------|------|
| 逐元素 1D | 闭包或模块级 | `block_size()` 在 `create_*_kernel()` 内或模块级定义 |
| 二维及以上 | **必须模块级** + 参数默认值 | `BLOCK_SIZE_M = block_size()` 在模块顶层 |
| 归约算子 | **必须模块级** + `constexpr=True` | `BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)` 在模块顶层 |

## 8. 常见问题

### 8.1 Tile 形状与 Tensor 维度不匹配

```python
# ❌ Tensor(1) 但 tile 用 2 个参数
x = Tensor(1)
x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

# ✅ Tensor(2) 配二参数 tile
x = Tensor(2)
x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))
```

### 8.2 归约算子用错 BLOCK_SIZE 类型

```python
# ❌ 归约用 block_size() 导致多 block 局部归约覆盖
BLOCK_SIZE = block_size()

# ✅ 归约必须用 constexpr
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
```

### 8.3 fp16 精度丢失

```python
# ❌ fp16 直接做 exp 可能精度不足
output = ntl.exp(x)

# ✅ 先转 fp32
x_fp32 = ntl.cast(x, ntl.float32)
output = ntl.exp(x_fp32)
```

### 8.4 调试：检查 generated source

```bash
# 列出最近生成的 Triton 源码
ls -lt ~/.ninetoothed/*.py | head -5

# 或通过 kernel.src 属性
python -c "from operators import create_add_kernel; k = create_add_kernel(); print(k.src)"
```

## 9. 参考资源

- **SKILL.md**：完整工作流、tile 决策速查表、编码模式、性能验证
- **operators/**：9 个已验证的九齿算子实现（最佳参考）
- **operator_patterns.md**：根据算子类型查找对应的 arrangement/application 模式
- **migration_guide.md**：Triton → 九齿的范式迁移指南
