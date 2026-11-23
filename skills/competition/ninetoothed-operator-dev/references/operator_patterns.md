# 九齿算子设计模式

本文档汇总了在 `operators/` 中已验证的算子设计模式。每个模式均来自可工作的代码，而非推测。

## 1. 三种核心 arrangement 模式

### 模式 A：逐元素（Elementwise）

所有维度用 `BLOCK_SIZE` 并行分块。

```python
# 1D 逐元素 — 参考 operators/add.py, relu.py, sigmoid.py, gelu.py
BLOCK_SIZE = block_size()  # 或 Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(x, output, BLOCK_SIZE=BLOCK_SIZE):
    return x.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

# 2D 逐元素 — 参考 operators/add_2d.py
BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()

def arrangement(x, y, z, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N):
    return (
        x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
        y.tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
        z.tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
    )
```

**关键约束**：
- 逐元素 1D 可用 `block_size()`（自动调优）或 `Symbol(constexpr=True)`
- 逐元素 2D **必须从模块级定义** Block Symbols，通过参数默认值传入

### 模式 B：归约（Reduction）

归约维度用 `BLOCK_SIZE`，保留维度用 `1`。

```python
# Softmax 沿列归约 — 参考 operators/softmax.py
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)  # ⚠️ 必须 constexpr！

def arrangement(x, output, BLOCK_SIZE=BLOCK_SIZE):
    # tile((1, BLOCK_SIZE)): 行维度完整(1)，列维度分块(BLOCK_SIZE)
    return x.tile((1, BLOCK_SIZE)), output.tile((1, BLOCK_SIZE))

def application(x, output):
    x_max = ntl.max(x)
    x_shifted = x - x_max
    exp_x = ntl.exp(x_shifted)
    sum_exp = ntl.sum(exp_x)
    output = exp_x / sum_exp  # noqa: F841

# Sum 全量归约 — 参考 operators/sum.py
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(x, output, BLOCK_SIZE=BLOCK_SIZE):
    return x.tile((BLOCK_SIZE,)), output.tile((1,))  # 输出为标量

def application(x, output):
    output = ntl.sum(x)  # noqa: F841
```

**关键约束**：
- **必须 `Symbol(constexpr=True)`**，不能用 `block_size()`。原因：`block_size()` 可能产生小于归约维度的值，导致多 block 各自独立归约，结果错误
- 调用时需传入完整维度值：`kernel(x, out, BLOCK_SIZE=x.shape[-1])`
- 返回标量时 `output.tile((1,))`（大小为 1 的 1D tile）

### 模式 C：非连续 / 步长（Strided）

在 tile 之前先做切片。

```python
# 1D stride=2 — 参考 operators/strided_add.py
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(input, other, output, BLOCK_SIZE=BLOCK_SIZE):
    return (
        input[::2].tile((BLOCK_SIZE,)),   # 切片先于 tile
        other[::2].tile((BLOCK_SIZE,)),
        output[::2].tile((BLOCK_SIZE,)),
    )

# 2D 沿列 stride=2 — 参考 operators/strided_add.py
BLOCK_SIZE_ROW = Symbol("BLOCK_SIZE_ROW", constexpr=True)
BLOCK_SIZE_COL = Symbol("BLOCK_SIZE_COL", constexpr=True)

def arrangement(input, other, output,
                BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
                BLOCK_SIZE_COL=BLOCK_SIZE_COL):
    return (
        input[:, ::2].tile((BLOCK_SIZE_ROW, BLOCK_SIZE_COL)),
        other[:, ::2].tile((BLOCK_SIZE_ROW, BLOCK_SIZE_COL)),
        output[:, ::2].tile((BLOCK_SIZE_ROW, BLOCK_SIZE_COL)),
    )
```

**关键约束**：
- 切片操作在 tile 之前
- 输出需预初始化为零（非步长位置不会被写入）
- stride≠1 导致非合并访存，性能约下降 15-27%（见 self_eval_4）

## 2. 融合算子模式（Reduction + Elementwise）

参考 `operators/rms_norm.py`。

```python
# RMS Norm: y = x * rsqrt(mean(x^2) + eps) * w
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(x, w, eps, y, BLOCK_SIZE=BLOCK_SIZE):
    def arrange(tensor):
        return tensor.tile((1, BLOCK_SIZE))
    return arrange(x), arrange(w), eps, arrange(y)  # eps 直接透传

def application(x, w, eps, y):
    x_fp32 = ntl.cast(x, ntl.float32)
    y = x_fp32 * ntl.rsqrt(
        ntl.sum(x_fp32 * x_fp32) / x.shape[-1] + eps
    ) * w  # noqa: F841

tensors = (Tensor(2), Tensor(2), Tensor(0), Tensor(2))
```

**融合算子的关键模式**：
1. **标量参数透传**：`eps` 用 `Tensor(0)` 定义，不加 tile，直接透传到 application
2. **多张量同 tile**：使用局部函数对多个同维张量应用相同 tiling
3. **混合精度**：`ntl.cast(x, ntl.float32)` 在 fp16 输入上进行 fp32 计算
4. **融合范围**：square → sum → divide → rsqrt → rescale × w 全在单个 kernel

**对比非融合 vs 融合**：

| 方面 | 非融合（多 kernel） | 融合（单 kernel） |
|------|-------------------|-------------------|
| 中间张量 | 需要多次 load/store | 寄存器内计算 |
| 访存次数 | N 次 | 1 次 |

## 3. 三种常见函数签名模式

### 3.1 单输入单输出

```python
def arrangement(input, output):
    ...
def application(input, output):
    output = some_op(input)  # noqa: F841
```

算子：relu, sigmoid, gelu

### 3.2 双输入单输出

```python
def arrangement(x, y, output):
    ...
def application(x, y, output):
    output = x + y  # noqa: F841
```

算子：add, add_2d, strided_add

### 3.3 多输入含标量

```python
def arrangement(x, w, eps, y):
    def arrange(t): return t.tile((1, BLOCK_SIZE))
    return arrange(x), arrange(w), eps, arrange(y)
```

算子：rms_norm

## 4. BLOCK_SIZE 选择指南

| 场景 | 使用 | 原因 |
|------|------|------|
| 逐元素 1D | `block_size()` 或 `Symbol(constexpr=True)` | 两种均可，`block_size()` 可自动调优 |
| 逐元素 2D+ | `block_size()`（模块级） | 自动调优多维 tile 大小 |
| 归约算子 | `Symbol(constexpr=True)`（模块级） | **必须**，防止多 block 局部归约覆盖 |
| 非连续/步长 | `Symbol(constexpr=True)` | 切片后有效大小需精确匹配 |

## 5. SKILL.md 中的 Tile 决策速查表

此为完整参考，以下摘录关键行：

| 算子类型 | Tensor | tile 形状 | BLOCK_SIZE 类型 |
|---------|--------|----------|----------------|
| 逐元素 1D | `Tensor(1)` | `(BLOCK_SIZE,)` | `block_size()` |
| 逐元素 2D | `Tensor(2)` | `(B_M, B_N)` | `block_size()` |
| 归约 2D 沿列 | `Tensor(2)` | `(1, BLOCK_SIZE)` | **必须** `Symbol(constexpr=True)` |
| 融合归约 | `Tensor(2)` | `(1, BLOCK_SIZE)` + 标量 | **必须** `Symbol(constexpr=True)` |
| 非连续 1D | `Tensor(1)` | `(BLOCK_SIZE,)` + slice | `Symbol(constexpr=True)` |

> 完整速查表见 SKILL.md §2.4。

## 6. 实际算子索引

以下 9 个算子是本 skill 中所有模式的**可执行参考实现**：

| 算子 | 文件 | 模式 | 关键知识点 |
|------|------|------|-----------|
| Add 1D | `add.py` | 逐元素 | 模块级 `block_size()` + 参数默认值 |
| Add 2D | `add_2d.py` | 逐元素 2D | 多维 `block_size()` 模块级定义 |
| ReLU | `relu.py` | 逐元素 + ntl.maximum | `ntl.maximum(x, 0.0)` |
| Sigmoid | `sigmoid.py` | 逐元素 + fp32 cast | `ntl.cast(x, ntl.float32)` |
| GELU | `gelu.py` | 逐元素 + sigmoid 恒等式 | `tanh(z)` = `2*sigmoid(2z)-1` |
| Softmax | `softmax.py` | 归约沿列 | `constexpr` + `(1, BLOCK_SIZE)` |
| Sum | `sum.py` | 全量归约 | 输出 `tile((1,))` |
| Strided Add | `strided_add.py` | 非连续 1D+2D | `[::2]` 切片 + tile |
| RMS Norm | `rms_norm.py` | 融合归约 | `Tensor(0)` 透传标量 |

> 编写新算子时，先在此表找到最接近的模式，再查看对应源代码。
