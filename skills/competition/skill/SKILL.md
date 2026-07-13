---
name: skill
description: 使用 ninetoothed DSL 开发 Triton 算子的完整指南，涵盖开发→测试→诊断→修复全流程。
---

---

## 0. 工作流总览（开发 → 测试 → 诊断 → 修复）

这是使用本 skill 包开发算子的完整生命周期。当收到开发算子的任务时，按以下四个阶段依次执行。

---

### Phase 1：开发算子

**步骤**：
1. **理解算子** — 分析算子的计算逻辑（element-wise / reduction / matmul / norm / attention 等）、输入输出关系、广播/归约操作
2. **选择 DSL 模式** — 对照下文 §4 Arrangement 模式和 §5 Application 模式，确定对应模板
3. **复用模板** — 从 `skill/templates/` 选择最接近的模板文件，复制到目标文件
4. **编写 arrangement** — 根据模式定义数据布局（tile/expand/squeeze 等）
5. **编写 application** — 使用 `ntl.*` API 编写块内计算逻辑（⚠️ 注意 AST 跟踪陷阱，见 §8）
6. **声明 Tensor 与 Symbol** — `Tensor(ndim)` 的 ndim 与实际张量维度一致
7. **创建 kernel** — 调用 `ninetoothed.make(arrangement, application, tensors)`
8. **编写 torch 包装层** — 参考 §7 的模式，建议使用 flatten_wrapper（非连续安全）

**参考材料**：
- 模式对照：`skill/references/dsl_patterns.md`
- 模板文件：`skill/templates/elementwise_1d.py`、`skill/templates/activation.py` 等

---

### Phase 2：测试算子

**编写测试**：在算子所在文件中添加测试套件，覆盖以下场景：

| 维度 | 场景 | 检查点 |
|------|------|--------|
| 基础正确性 | contiguous 张量，标准 shape | 与 `torch.*` 或 `F.*` 输出 allclose |
| 半精度 | fp16 / bf16 输入 | atol=1e-3, rtol=1e-3 |
| 广播 | expand_as 创建 stride=0 视图 | 正确广播 |
| 非连续 | `.t()` 转置、`[::2]` 切片 | 数值正确，不 crash |

**运行测试（Linux）**：

```bash
# Option A：直接运行（推荐开发阶段使用）
python path/to/your_operator_file.py

# Option B：使用 test 脚本
bash skill/scripts/run_correctness.sh -f path/to/your_file.py

# Option C：运行 skill 全部测试
bash skill/scripts/run_correctness.sh
```

**参考材料**：
- 测试模式：`skill/references/testing_patterns.md`
- 验证脚本：`skill/scripts/validate_skill_package.py`

---

### Phase 3：诊断失败

当测试报错或结果不正确时，按以下顺序排查：

**1. 编译错误（Crash / CUDA Error / NameError）**

常见症状与速查：

| 报错信息 | 最可能原因 | 解决方案 |
|----------|-----------|---------|
| `NameError: name 'math' is not defined` | application 中引用了 `math.*` | 用字面量数值替代 |
| `NameError: name 'XXX' is not defined` | 模块级变量被 AST 跟踪 | 将常量值内联到 application |
| `AttributeError: __pow__` | 使用了 `**` 运算符 | 用 `x * x * x` 替代 |
| `AttributeError: tanh` | 调用了 `ntl.tanh` | 用 `ntl.exp` 手动实现 |
| `make()` 编译失败 | Tensor 声明/符号不匹配 | 检查 Tensor ndim 和 Symbol 传递 |

**2. 正确性失败（数值不匹配）**

- **检查 dtype** — 是否需要 `ntl.cast` 到 float32 计算？
- **检查 BLOCK_SIZE** — 是否整除总元素数？不能整除时需要 mask
- **检查非连续张量** — arrangement 是否用了 `flatten()`？→ 改为 preserve-ndim tile
- **检查广播** — 广播维度是否正确 expand？

**3. 诊断工具**

```bash
# 查看生成的 Triton 源码（用于诊断 AST 嵌入问题）
bash skill/scripts/inspect_generated_source.sh your_operator_file.py

# 查看日志
python skill/scripts/collect_task_log.py --output diagnose_log/
```

**参考材料**：
- 故障诊断指南：`skill/references/failure_diagnosis.md`
- 诊断模板：`skill/templates/failure_diagnosis_template.md`
- 生成源码检查：`skill/references/generated_source_and_aot.md`

---

### Phase 4：修复并回归

1. 根据诊断结果修改代码
2. **优先使用编辑工具** 进行定点修改（`replace_string_in_file` / `multi_replace_string_in_file`）
3. 重新运行测试（回到 Phase 2）
4. 确认全部测试通过后，填写算子任务报告：

```bash
# 参考模板记录实现过程
cat skill/templates/operator_task_report_template.md
```

---

### 全流程示例（以 ReLU 为例）

```bash
# 1. 开发 — 在 basic_operators.py 中实现 make_relu()
#    参考 skill/templates/elementwise_1d.py 模板

# 2. 测试 — 运行测试
python basic_operators.py
#    输出: ✅ ReLU 测试通过

# 3. 诊断（失败时）
#    报错: AssertionError: ReLU non-contiguous failed
#    根因: arrangement 用 flatten() 破坏了 strides
#    修复: tile_shape = (1,)*(ndim-1) + (block_size,)，使用 preserve-ndim tile

# 4. 回归
python basic_operators.py
#    输出: 🎉 所有测试全部通过！
```

---

## 1. 核心概念

**注意**引入常用的九齿python包
```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor, block_size

```

### 1.1 DSL 三层架构

每个 ninetoothed 算子由三个部分组成：

```
┌─────────────────────────────────────────┐
│   torch 包装层                           │
│   (预处理/后处理、shape 调整、调用 kernel) │
├─────────────────────────────────────────┤
│   arrangement 函数                       │
│   (数据布局、分块 tile、维度变换)          │
├─────────────────────────────────────────┤
│   application 函数                       │
│   (逐块计算逻辑，使用 ntl 语言)            │
└─────────────────────────────────────────┘
```

- **torch 包装层**：负责创建 output tensor、reshape/flatten、调用 `kernel()` 并传参
- **arrangement**：描述每个 Tensor 如何被分成 tile 块，以及块间维度关系
- **application**：定义每个 tile 内的计算逻辑（对标 Triton kernel 体）

### 1.2 基本流程

```python
import ninetoothed
from ninetoothed import Symbol, Tensor

# 1. 定义符号（block 大小等）
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

# 2. 定义数据布局
def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

# 3. 定义计算逻辑
def application(input, output):
    output = input * 2  # noqa: F841

# 4. 声明 tensor 元信息
tensors = (Tensor(1), Tensor(1))

# 5. 创建 kernel
kernel = ninetoothed.make(arrangement, application, tensors)
```

---

## 2. Symbol —— 符号参数系统

### 2.1 符号类型

```python
from ninetoothed import Symbol, block_size

# constexpr —— 编译时常量，在 make() 时确定
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

# meta —— 元参数，Triton autotune 可自动搜索
BLOCK_SIZE_M = block_size()                    # 等价于 Symbol("BLOCK_SIZE_M", meta=True)

# constexpr + upper_bound —— 常量且有上界，用于窗口/卷积
WINDOW_HEIGHT = Symbol("WINDOW_HEIGHT", constexpr=True, upper_bound=16)
WINDOW_WIDTH  = Symbol("WINDOW_WIDTH", constexpr=True, upper_bound=16)

# constexpr + upper_bound —— 用于 scale 等标量
SCALE = Symbol("SCALE", constexpr=True, upper_bound=128)
```

### 2.2 符号传递方式

```python
# 方式1：在 kernel() 调用时传参（适用于 constexpr）
kernel(input, other, output, BLOCK_SIZE=1024)

# 方式2：由 autotune 自动搜索（适用于 meta/block_size）
# 无需手动传值
```

### 2.3 符号命名约定

| 符号名 | 用途 | 推荐类型 |
|--------|------|----------|
| `BLOCK_SIZE` | 1D 通用 tile 大小 | `constexpr=True` |
| `BLOCK_SIZE_M` | 矩阵 M 维度 tile | `block_size()` |
| `BLOCK_SIZE_N` | 矩阵 N 维度 tile | `block_size()` |
| `BLOCK_SIZE_K` | 矩阵 K 维度 tile | `block_size()` |
| `WINDOW_HEIGHT` | 池化/卷积窗口高度 | `constexpr=True, upper_bound=N` |
| `WINDOW_WIDTH` | 池化/卷积窗口宽度 | `constexpr=True, upper_bound=N` |

---

## 3. Tensor —— 张量元信息声明

### 3.1 Tensor 构造函数

```python
Tensor(ndim, other=, shape_options=)
```

| 参数 | 说明 |
|------|------|
| `ndim` | 张量维度数（int） |
| `other=float("-inf")` | 边界填充值（用于 max 或 softmax 等归约操作） |
| `shape_options=None / dict` | 形状选项，如 `{"constexpr": True}` 或 `{"constexpr": True, "upper_bound": 128}` |

### 3.2 常见 Tensor 声明

```python
# 标量
Tensor(0)                              # 0 维张量（eps, beta, alpha, scale）

# 1D 张量
Tensor(1)                              # 1 维，用于 element-wise

# 2D 张量
Tensor(2)                              # 2 维矩阵
Tensor(2, other=float("-inf"))         # 2D + 负无穷边界（softmax）

# 3D 张量
Tensor(3)                              # 3D（bmm 等）

# 4D 张量
Tensor(4)                              # 4D（conv2d, rope, attention）
Tensor(4, shape_options={"constexpr": True})  # 部分维度 constexpr

# 批量 Tensor 声明
tuple(Tensor(4, shape_options=shape_options) for _ in range(3))
```

### 3.3 Tensor 元组规范

`tensors` 元组的顺序必须与 `arrangement` 和 `application` 的参数顺序**完全一致**。

```python
# add: 3 个 1D tensor
tensors = tuple(Tensor(1) for _ in range(3))

# softmax: 2D input (边界 -inf) + 2D output
tensors = (Tensor(2, other=float("-inf")), Tensor(2))

# mm: 3 个 2D tensor
tensors = (Tensor(2), Tensor(2), Tensor(2))

# fused_rms_norm: 两个 2D + 一个标量 + 一个 2D
tensors = (Tensor(2), Tensor(2), Tensor(0), Tensor(2))
```

---

## 4. Arrangement —— 数据布局模式大全

### 4.1 基础 tile 模式

#### 模式 A：简单 1D tile（add, silu, swiglu）

所有参与张量沿最后一维均匀分块：

```python
def arrangement(input, other, output, BLOCK_SIZE=BLOCK_SIZE):
    input_arranged  = input.tile((BLOCK_SIZE,))
    other_arranged  = other.tile((BLOCK_SIZE,))
    output_arranged = output.tile((BLOCK_SIZE,))
    return input_arranged, other_arranged, output_arranged
```

#### 模式 A2：非连续安全的 Multi-ND tile（推荐用于 element-wise 算子）

和模式 A 不同，本模式**不进行 flatten**，保留原始 strides，确保 `.T` / `.t()` 等非连续张量的读写正确：

```python
def _element_wise_arrangement(*tensors, block_size):
    ndim = max(tensor.ndim for tensor in tensors)
    assert all(tensor.ndim == ndim or tensor.ndim == 0 for tensor in tensors)
    tile_shape = tuple(1 for _ in range(ndim - 1)) + (block_size,)
    return tuple(
        tensor.tile(tile_shape) if tensor.ndim != 0 else tensor
        for tensor in tensors
    )
```

| 区别 | 模式 A（flatten） | 模式 A2（非 flatten） |
|------|-------------------|----------------------|
| 张量声明 | `Tensor(1)` | `Tensor(ndim)`（与实际 ndim 一致） |
| 非连续支持 | ❌ flatten 破坏 strides | ✅ 保留原始 strides |
| 标量支持 | 需额外处理 | ✅ `tensor.ndim != 0` 自动跳过 |

#### 模式 B：2D 行 tile（softmax, rms_norm）

保留第一维（batch），分块第二维：

```python
def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((1, BLOCK_SIZE)), output.tile((1, BLOCK_SIZE))
```

#### 模式 C：2D 分块 Matmul（mm）

核心技巧：先将 output 按 `(BLOCK_SIZE_M, BLOCK_SIZE_N)` 分块，再让 input 和 other 通过 `tile + expand + squeeze` 与其对齐。

```python
def arrangement(input, other, output,
                BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K):

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
```

#### 模式 D：3D 分块 Matmul（bmm）

在 mm 前加一个 `tile((1, ...))` 保留 batch 维度：

```python
def arrangement(
    input, other, output,
    BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K
):
    output_arranged = output.tile((1, BLOCK_SIZE_M, BLOCK_SIZE_N))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    input_arranged = input.tile((1, BLOCK_SIZE_M, BLOCK_SIZE_K))
    input_arranged = input_arranged.tile((1, 1, -1))
    input_arranged = input_arranged.expand((-1, -1, output_arranged.shape[-1]))
    input_arranged.dtype = input_arranged.dtype.squeeze((0, 1))
    input_arranged.dtype.dtype = input_arranged.dtype.dtype.squeeze(0)

    other_arranged = other.tile((1, BLOCK_SIZE_K, BLOCK_SIZE_N))
    other_arranged = other_arranged.tile((1, -1, 1))
    other_arranged = other_arranged.expand((-1, output_arranged.shape[-2], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze((0, 2))
    other_arranged.dtype.dtype = other_arranged.dtype.dtype.squeeze(0)

    return input_arranged, other_arranged, output_arranged
```

#### 模式 E：自定义 strides/dilation tile（RoPE）

用于非连续内存访问，如 Rotary Position Embedding 中的交错/非交错模式：

```python
def arrangement(input, sin_table, cos_table, interleaved=True):
    emb_dim = input.shape[-1]
    tile_shape = (1, 1, 1, emb_dim // 2)

    if interleaved:
        strides = (-1, -1, -1, 1)
        dilation = (1, 1, 1, 2)
    else:
        strides = None
        dilation = None

    input_arranged = input.tile(tile_shape, strides=strides, dilation=dilation)
    input_arranged = input_arranged.tile((1, 1, 1, 2))
    input_arranged.dtype = input_arranged.dtype.squeeze((0, 1, 2))
    input_arranged.dtype.dtype = input_arranged.dtype.dtype.squeeze((0, 1, 2))

    sin_table_arranged = sin_table.tile(tile_shape)
    sin_table_arranged.dtype = sin_table_arranged.dtype.squeeze((0, 1, 2))

    cos_table_arranged = cos_table.tile(tile_shape)
    cos_table_arranged.dtype = cos_table_arranged.dtype.squeeze((0, 1, 2))

    return input_arranged, sin_table_arranged, cos_table_arranged
```

#### 模式 F：窗口 + ravel + flatten（max_pool2d）

先 tile 窗口，再 ravel+flatten 将窗口内元素展平，最后 tile 做 block：

```python
def arrangement(input, output):
    input_arranged = input.tile((1, 1, WINDOW_HEIGHT, WINDOW_WIDTH))
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=4).flatten(start_dim=1)
    input_arranged = input_arranged.tile((BLOCK_SIZE, -1))

    output_arranged = output.tile((1, 1, 1, 1))
    output_arranged = output_arranged.ravel()
    output_arranged = output_arranged.flatten(end_dim=4).flatten(start_dim=1)
    output_arranged = output_arranged.tile((BLOCK_SIZE, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)

    return input_arranged, output_arranged
```

#### 模式 G：复用作现有 arrangement（addmm）

```python
# addmm —— 复用 mm.arrangement，额外传 input/beta/alpha
def arrangement(input, mat1, mat2, beta, alpha, output):
    _, _, input_arranged = mm.arrangement(mat1, mat2, input)
    mat1_arranged, mat2_arranged, output_arranged = mm.arrangement(mat1, mat2, output)
    return input_arranged, mat1_arranged, mat2_arranged, beta, alpha, output_arranged
```

#### 模式 H：conv2d —— im2col 式 flatten + 复用 mm

```python
def arrangement(input, filter, output):
    input_arranged = input.tile((1, *filter.shape[1:]), strides=(-1, -1, 1, 1))
    input_arranged = input_arranged.squeeze(1)
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=3).flatten(start_dim=1)

    filter_arranged = filter.flatten(start_dim=1)
    filter_arranged = filter_arranged.permute((1, 0))

    output_arranged = output.permute((0, 2, 3, 1)).flatten(end_dim=3)

    return mm.arrangement(input_arranged, filter_arranged, output_arranged)
```

#### 模式 I：Attention —— online softmax

```python
def arrangement(
    q, k, v, scale, o, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
):
    def arrange_q_or_o(input):
        arranged = input.tile((1, 1, BLOCK_SIZE_M, -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1))
        return arranged

    def arrange_k_or_v(input):
        arranged = input.tile((1, 1, BLOCK_SIZE_N, -1))
        arranged = arranged.tile((1, 1, -1, -1))
        arranged = arranged.expand((-1, -1, q_arranged.shape[-2], -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
        return arranged

    q_arranged = arrange_q_or_o(q)
    return q_arranged, arrange_k_or_v(k), arrange_k_or_v(v), scale, arrange_q_or_o(o)
```

### 4.2 Arrangement 方法速查

| 方法 | 作用 | 示例 |
|------|------|------|
| `.tile(shape)` | 按 shape 分块 | `tensor.tile((BLOCK_SIZE,))` |
| `.tile(shape, strides=..., dilation=...)` | 自定义步幅/膨胀分块 | RoPE 交错访问 |
| `.expand(shape)` | 广播扩展维度（类似 torch.expand） | mm 中 K 维对齐 |
| `.squeeze(dim)` | 在 arrangement 的 dtype 上下文删除维度 | `dtype.squeeze(0)` |
| `.ravel()` | 将连续内存 tile 展平为一维 | pool2d 窗口展平 |
| `.flatten(start_dim, end_dim)` | 展平指定维度范围 | conv2d 的 im2col |
| `.permute(axes)` | 重排维度顺序 | conv2d filter/output 变换 |

### 4.3 `.dtype` 操作详解

`arranged_tensor.dtype` 是一个代理对象，用来在符号层面描述数据类型和维度关系。

```python
# 删掉某个轴（该轴大小变为 1 时有效）
arranged.dtype = arranged.dtype.squeeze(0)
arranged.dtype = arranged.dtype.squeeze((0, 1))

# 进一步操作深层 dtype（多重 tile/expand 后）
arranged.dtype.dtype = arranged.dtype.dtype.squeeze(0)
arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1, 2))
```

---

## 5. Application —— 计算逻辑模式大全

### 5.1 标量算术模式

```python
# add —— 逐元素加法
def application(input, other, output):
    output = input + other  # noqa: F841
```

### 5.2 激活函数模式

```python
# silu: x * sigmoid(x)
def application(input, output):
    input_loaded = input
    output = input_loaded * ntl.sigmoid(ntl.cast(input_loaded, ntl.float32))  # noqa: F841

# swiglu: a * (b * sigmoid(b))
def application(a, b, c):
    b_loaded = b
    gate = b_loaded * ntl.sigmoid(ntl.cast(b_loaded, ntl.float32))
    c = a * gate  # noqa: F841
```

# GELU approximate（tanh 近似，用 ntl.exp 手动实现 tanh）
```python
def application(input, output):
    # tanh 近似: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    # 所有数值必须用字面量！不能引用 math.pi 或模块级变量（AST 跟踪陷阱）
    t = 0.7978845608028654 * (input + 0.044715 * input * input * input)
    exp_t = ntl.exp(t)
    exp_neg_t = ntl.exp(-t)
    output = 0.5 * input * (1.0 + (exp_t - exp_neg_t) / (exp_t + exp_neg_t))  # noqa: F841

# GELU exact（标准 erf 公式）
```python
def application(input, output):
    output = input * 0.5 * (1.0 + ntl.erf(input / ntl.sqrt(2.0)))  # noqa: F841
```

**GELU 注意事项**：
- `x ** 3` 不可用（Triton tensor 不支持 `__pow__`） → 用 `x * x * x`
- `ntl.sqrt(2.0 / math.pi)` 不可用 → 用字面量 `0.7978845608028654`
- `ntl.tanh` 不可用 → 用 `(exp(t)-exp(-t))/(exp(t)+exp(-t))`
- 测试对比：`torch.nn.functional.gelu(x, approximate='tanh')`（近似版）和 `torch.nn.functional.gelu(x)`（精确版）

### 5.3 归约模式

```python
# softmax: online softmax（行级归约）
def application(input, output):
    input_loaded = input
    row_minus_max = input_loaded - ntl.max(input_loaded)
    numerator = ntl.exp(row_minus_max)
    denominator = ntl.sum(numerator)
    output = numerator / denominator  # noqa: F841

# max_pool2d: 窗口内 max
def application(input, output):
    output = ntl.max(input, axis=1)  # noqa: F841
```

### 5.4 矩阵乘模式

```python
# mm —— 累加 dot product
def application(input, other, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(input.shape[0]):
        accumulator += ntl.dot(input[k], other[k])
    output = accumulator

# bmm —— 直接复用 mm.application
application = mm.application
```

### 5.5 归一化模式

```python
# rms_norm: x / sqrt(mean(x^2) + eps)
def application(input, eps, output):
    input_fp32 = ntl.cast(input, ntl.float32)
    output = input_fp32 * ntl.rsqrt(  # noqa: F841
        ntl.sum(input_fp32 * input_fp32) / input.shape[-1] + eps
    )

# fused_rms_norm: (x / sqrt(mean(x^2) + eps)) * w
def application(x, w, eps, y):
    x_fp32 = ntl.cast(x, ntl.float32)
    y = x_fp32 * ntl.rsqrt(ntl.sum(x_fp32 * x_fp32) / x.shape[-1] + eps) * w  # noqa: F841
```

### 5.6 Attention 模式（Online Softmax Flash Attention）

```python
def application(q, k, v, scale, o):
    # 变换 Q 使其与 K 的乘积可表示为 exp2 形式
    q_loaded = (q * scale * 1.44269504089).to(q.dtype)  # log2(e)
    acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
    l_i = ntl.full((q.shape[-2],), 1, dtype=ntl.float32)
    m_i = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)

    for i in range(k.shape[0]):
        qk = ntl.dot(q_loaded, ntl.trans(k[i]))
        qk = ntl.where(k[i].offsets(-2) < k.source.shape[-2], qk, float("-inf"))
        m_ij = ntl.maximum(m_i, ntl.max(qk, 1))
        p = ntl.exp2(qk - m_ij[:, None])
        l_ij = ntl.sum(p, 1)
        alpha = ntl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + ntl.dot(p.to(v.dtype.dtype), v[i])
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    acc /= l_i[:, None]
    o = acc.to(o.dtype)  # noqa: F841
```

### 5.7 RoPE 模式（索引级写）

```python
def application(input, sin_table, cos_table):
    sin_table_loaded = sin_table
    cos_table_loaded = cos_table
    input_0 = input[0]
    input_1 = input[1]
    input[0] = input_0 * cos_table_loaded - input_1 * sin_table_loaded
    input[1] = input_0 * sin_table_loaded + input_1 * cos_table_loaded
```

### 5.8 Composite 模式（复用现有 application）

```python
# addmm —— 先用 mm 算 matmul，再加 bias+scale
def application(input, mat1, mat2, beta, alpha, output):
    mm.application(mat1, mat2, output)
    output = beta * input + alpha * output
```

### 5.9 application 注意事项

- **末尾必须用 `# noqa: F841`** 标注赋值，因为 Triton 在汇编阶段才会实际使用变量
- `x.shape`、`x.dtype` 在 application 中是符号表达式，不是具体数值/类型
- `x.source` 可访问到原始未 tile 的张量元信息
- `.offsets(dim)` 返回当前块在指定维度的起始偏移量
- 所有运算都是**符号化**的，最终由 Triton JIT 编译为 GPU 代码

---

## 6. Kernel 创建与调用

### 6.1 标准创建

```python
kernel = ninetoothed.make(arrangement, application, tensors)
```

### 6.2 多 kernel 分支（RoPE）

```python
interleaved_kernel = ninetoothed.make(
    functools.partial(arrangement, interleaved=True), application, inputs
)
non_interleaved_kernel = ninetoothed.make(
    functools.partial(arrangement, interleaved=False), application, inputs
)

def kernel(input, sin_table, cos_table, interleaved=True):
    return (interleaved_kernel if interleaved else non_interleaved_kernel)(
        input, sin_table, cos_table
    )
```

### 6.3 调用 kernel

```python
# constexpr 符号作为 kwargs 传入
kernel(input, other, output, BLOCK_SIZE=1024)

# meta/block_size 符号无需传入，由 autotune 自动搜索
kernel(mat1, mat2, output)
```

---

## 7. Torch 包装层模式

### 7.1 Element-wise（add, silu）

```python
def add(input, other):
    output = torch.empty_like(input)
    ops.ninetoothed.kernels.add.kernel(input, other, output, BLOCK_SIZE=1024)
    return output
```

### 7.2 Flatten 后调用（silu, swiglu）

```python
def silu(input):
    input_flat = input.flatten()
    output_flat = torch.empty_like(input_flat)
    ops.ninetoothed.kernels.silu.kernel(input_flat, output_flat, BLOCK_SIZE=1024)
    return output_flat.view_as(input)
```

### 7.3 Matmul（mm, bmm, addmm）

```python
def mm(input, other):
    output_shape = (input.shape[0], other.shape[1])
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    ops.ninetoothed.kernels.mm.kernel(input, other, output)
    return output
```

### 7.4 Norm（rms_norm, fused_rms_norm）

```python
def fused_rms_norm(x, w, eps=None):
    if eps is None:
        eps = torch.finfo(x.dtype).eps()
    x_2d = x.view(-1, x.shape[-1])
    w_2d = w.expand_as(x_2d)
    y_2d = torch.empty_like(x_2d)
    ops.ninetoothed.kernels.fused_rms_norm.kernel(x_2d, w_2d, eps, y_2d, BLOCK_SIZE=x.shape[-1])
    return y_2d.view(x.shape)
```

### 7.5 Conv2d

```python
def conv2d(input, filter):
    n, _, h, w = input.shape
    k, _, r, s = filter.shape
    p, q = h - r + 1, w - s + 1
    output = torch.empty((n, k, p, q), dtype=input.dtype, device=input.device)
    ops.ninetoothed.kernels.conv2d.kernel(input, filter, output)
    return output
```

### 7.6 Attention

```python
def scaled_dot_product_attention(q, k, v, scale=None):
    if scale is None:
        scale = 1 / math.sqrt(q.shape[-1])
    o = torch.empty_like(q)
    ops.ninetoothed.kernels.scaled_dot_product_attention.kernel(q, k, v, scale, o)
    return o
```

### 7.7 Pooling

```python
def max_pool2d(input, window_shape):
    n, c, h, w = input.shape
    r, s = window_shape
    p = math.ceil((h - r) / r + 1)
    q = math.ceil((w - s) / s + 1)
    output = torch.empty(n, c, p, q, dtype=input.dtype, device=input.device)
    ops.ninetoothed.kernels.max_pool2d.kernel(input, output, WINDOW_HEIGHT=r, WINDOW_WIDTH=s)
    return output
```

---

## 8. AST 跟踪陷阱（Nineteethed DSL 特有）

Nineteethed 使用 Python AST 跟踪来生成 Triton 代码。`application()` 函数中**出现的所有 Python 变量名都会被原样嵌入生成的 Triton 代码**，而 Triton 编译环境中没有标准 Python 模块。

### 8.1 常见错误模式

```python
# ❌ 错误：math 模块在 Triton 中不存在
def application(input, output):
    t = ntl.sqrt(2.0 / math.pi) * input  # NameError: name 'math' is not defined

# ❌ 错误：模块级变量也被嵌入
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
def application(input, output):
    t = _SQRT_2_OVER_PI * input  # NameError: name '_SQRT_2_OVER_PI' is not defined

# ❌ 错误：__pow__ 在 Triton tensor 上不存在
def application(input, output):
    t = input ** 3  # AttributeError: 'tensor' object has no attribute '__pow__'

# ❌ 错误：ntl.tanh 不存在
def application(input, output):
    output = ntl.tanh(input)  # AttributeError
```

### 8.2 正确做法

```python
# ✅ 正确：使用字面量数值，inline 所有常量
def application(input, output):
    t = 0.7978845608028654 * (input + 0.044715 * input * input * input)
    exp_t = ntl.exp(t)
    exp_neg_t = ntl.exp(-t)
    output = 0.5 * input * (1.0 + (exp_t - exp_neg_t) / (exp_t + exp_neg_t))  # noqa: F841
```

### 8.3 规则总结

| 规则 | 说明 |
|------|------|
| ✅ 允许 | 字面量数值（`1.0`, `0.5`）、`ntl.*` API、基础运算符（`+`, `-`, `*`, `/`） |
| ❌ 禁止 | `math.*`、`torch.*`、模块级 Python 变量、`**` 运算符、NumPy 函数 |
| 🔧 替代方案 | 字面量 inline、`x * x * x` 替代 `x ** 3`、`ntl.exp` 组合替代 `ntl.tanh` |

---

## 9. ntl 语言速查

```python
import ninetoothed.language as ntl
```

### 8.1 类型转换

| API | 说明 |
|-----|------|
| `ntl.cast(x, ntl.float32)` | 显式类型转换 |
| `x.to(new_dtype)` | 隐式类型转换 |
| `ntl.float32` | float32 类型常量 |

### 8.2 数学运算

| API | 说明 |
|-----|------|
| `ntl.sigmoid(x)` | Sigmoid 激活函数 |
| `ntl.exp(x)` | 自然指数 |
| `ntl.exp2(x)` | 2 的幂 |
| `ntl.rsqrt(x)` | 1/sqrt(x) |
| `ntl.max(x)` | 最大值（单输入） |
| `ntl.max(x, axis=N)` | 沿指定轴最大值 |
| `ntl.maximum(a, b)` | 逐元素最大值（二元） |
| `ntl.sum(x)` | 求和 |
| `ntl.sum(x, axis=N)` | 沿轴求和 |
| `ntl.dot(a, b)` | 矩阵乘法（二维块级） |
| `ntl.trans(x)` | 转置 |
| `ntl.where(cond, x, y)` | 条件选择 |

### 8.3 初始化/创建

| API | 说明 |
|-----|------|
| `ntl.zeros(shape, dtype)` | 全零（在 application 内初始化累加器） |
| `ntl.full(shape, value, dtype)` | 填充指定值 |

### 8.4 张量元信息

| 属性 | 说明 |
|------|------|
| `x.shape` | tile 之后的局部形状 |
| `x.dtype` | tile 之后的数据类型 |
| `x.source` | 原始（未 tile）张量元信息 |
| `x.offsets(dim)` | 当前块在 dim 维度的起始偏移 |

---

## 10. 常见模式速查

### 10.1 1D Element-wise 模式

```python
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))
def application(input, output):
    output = <FORMULA>  # noqa: F841
tensors = (Tensor(1), Tensor(1))
kernel = ninetoothed.make(arrangement, application, tensors)
```

### 10.2 2D 行归约模式

```python
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((1, BLOCK_SIZE)), output.tile((1, BLOCK_SIZE))
def application(input, output):
    output = <REDUCTION>  # noqa: F841
tensors = (Tensor(2, other=float("-inf")), Tensor(2))
kernel = ninetoothed.make(arrangement, application, tensors)
```

### 10.3 Matmul 模式

```python
BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()
BLOCK_SIZE_K = block_size()
# ... mm arrangement + 循环 dot
tensors = (Tensor(2), Tensor(2), Tensor(2))
kernel = ninetoothed.make(arrangement, application, tensors)
```

### 9.4 复用模式

```python
# 复用 application
from somewhere import application as base_application
tensors = ...
kernel = ninetoothed.make(arrangement, base_application, tensors)

# 复用 arrangement + application
from somewhere import arrangement as base_arrangement, application as base_application
tensors = ...
kernel = ninetoothed.make(base_arrangement, base_application, tensors)
```

### 9.5 包装层 flatten 模式

```python
def op(input):
    flat = input.flatten()
    out_flat = torch.empty_like(flat)
    kernel(flat, out_flat, BLOCK_SIZE=1024)
    return out_flat.view_as(input)
```

### 9.6 标量参数传递

```python
# Tensor(0) 标量在 arrangement 中直接返回
def arrangement(x, eps, y):
    return x.tile((1, BLOCK_SIZE)), eps, y.tile((1, BLOCK_SIZE))

# kernel 调用时直接传值
kernel(input, eps_value, output, BLOCK_SIZE=1024)
```

---

## 11. 完整的算子开发工作流

### 步骤 1：创建 kernel 文件

```
ops/ninetoothed/kernels/my_op.py
```

### 步骤 2：实现 arrangement + application

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

def application(input, output):
    output = ntl.sigmoid(input)  # noqa: F841

tensors = (Tensor(1), Tensor(1))
kernel = ninetoothed.make(arrangement, application, tensors)
```

### 步骤 3：添加 torch 包装

在 `ops/ninetoothed/torch.py` 中添加：

```python
def my_op(input):
    flat = input.flatten()
    out_flat = torch.empty_like(flat)
    ops.ninetoothed.kernels.my_op.kernel(flat, out_flat, BLOCK_SIZE=1024)
    return out_flat.view_as(input)
```

### 步骤 4：测试

```python
import torch
from ops.ninetoothed.torch import my_op

x = torch.randn(4, 128, device="cuda")
result = my_op(x)
expected = torch.sigmoid(x)
assert torch.allclose(result, expected, atol=1e-5)
print("PASS")
```

---

## 12. 调试技巧

1. **检查 tile 形状**：在 arrangement 中打印 `x.shape` 观察分块是否符合预期
2. **dtype 操作链**：多层 `tile/expand` 后可能需要 `squeeze` 多次才能恢复正确的 dtype 结构
3. **noqa: F841**：application 中每个赋值语句都需要标注，否则 Python 字节码分析会警告
4. **`x.source`**：当需要引用原始张量信息时使用，例如 attention 中的 mask 边界检查
5. **`offsets(dim)`**：用于在 application 中判断当前块是否越界（如 attention 中 `k[i].offsets(-2) < k.source.shape[-2]`）
6. **BLOCK_SIZE 选择策略**：
   - element-wise：256~1024
   - matmul：M/N 64~128，K 32~64 （通常由 autotune 自动搜索）
   - softmax/reduction：取 `input.shape[-1]` 作为 BLOCK_SIZE（覆盖整行）
7. **测试时注意非连续张量**：如果包装层没有显式 `.contiguous()`，需确保 arrangement 正确处理 stride
