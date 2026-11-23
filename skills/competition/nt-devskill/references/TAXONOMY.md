# NineToothed 算子分类路由 (Taxonomy)

本文档将算子按计算模式分为六大族，用于指导 AI 在接收到算子开发请求时选择正确的实现策略和模板。

## 路由决策树

```
用户请求
├── 是否为已有算子的变体/组合？ → 查阅「组合复用」策略
├── elementwise（逐元素）？
│   ├── 一元 (unary)  → silu, gelu, relu, rsqrt, cast ...
│   └── 二元 (binary) → add, mul, sub, div ...
├── reduction（归约）？
│   ├── 单轴归约      → softmax, sum, max ...
│   └── 多轴归约      → layernorm, rms_norm ...
├── matmul-like（矩阵乘类）？
│   ├── 2D matmul     → mm, matmul ...
│   ├── batched       → bmm, addmm ...
│   └── fused         → linear+relu, addmm ...
├── attention（注意力）？
│   ├── SDPA          → scaled_dot_product_attention ...
│   └── multi-head    → attention module ...
├── convolution（卷积）？
│   └── conv2d        → im2col + matmul 复用 ...
├── generator（生成器）？
│   └── 从索引生成值   → linspace, arange, eye, meshgrid ...
├── layout/view（布局/视图操作）？ ← ⚠️ 注意：主操作是 view，但 .contiguous() 路径需要 kernel
│   ├── permute 类    → moveaxis, movedim, permute, transpose ...
│   ├── 切片类        → narrow, slice, split ...
│   └── 重排类        → channel_shuffle, scatter, reshape ...
│   → 实现策略：LIMITATION REPORT（主操作 view-only）+ Pattern 13 copy kernel（物化路径）
├── scatter（散布/索引写入）？
│   ├── index write     → slice_scatter, index_copy, scatter_add ...
│   └── permute write   → channel_shuffle, scatter ...
└── 其他 → 自定义算子（参考 OPTIMIZATION_GUIDE.md）
```

## 算子族详情

### 1. Elementwise — 逐元素算子

| 子类型   | 示例                     | 模板来源                  |
| -------- | ------------------------ | ------------------------- |
| 一元激活 | silu, gelu, relu, swiglu | `examples/silu/kernel.py` |
| 二元运算 | add, mul, sub, div       | `examples/add/kernel.py`  |
| 类型转换 | cast, to                 | —                         |

**实现策略：**
- Tile 模式：1D tiling，`BLOCK_SIZE=1024` 或 `block_size()`
- 模板：1D arrangement + 简单 expression
- 多维输入：先 flatten 为 1D，调用 kernel，再 reshape 回原形状（在 torch wrapper 中处理）
- 精度：对 sigmoid/rsqrt 等先用 `ntl.cast(x, ntl.float32)` 再计算

**公式速查：**

| 算子   | 公式                                                  |
| ------ | ----------------------------------------------------- |
| SiLU   | `x * sigmoid(x)`                                      |
| SwiGLU | `a * (b * sigmoid(b))`                                |
| GELU   | `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))` |
| ReLU   | `maximum(x, 0)`                                       |

**⚠️ 注意：** 当用户请求复杂激活函数（如 GELU 的近似公式）时，先确认数学公式再注入代码，不要猜测。

---

### 2. Reduction — 归约算子

| 子类型   | 示例                      | 模板来源                            |
| -------- | ------------------------- | ----------------------------------- |
| 单轴归约 | softmax, sum, max, min    | `examples/softmax/kernel.py`        |
| 归一化   | fused_rms_norm, layernorm | `examples/fused_rms_norm/kernel.py` |

**实现策略：**
- Tile 模式：2D tiling `(1, BLOCK_SIZE)`，一次处理一行
- 数值稳定性：`Tensor(other=float("-inf"))` 用于 max 的 partial tile 填充
- 精度：用 `ntl.cast` 转 float32 做累加，最终 `.to(dtype)` 写回
- BLOCK_SIZE 建议设为 `input.shape[-1]`（整行 fit 在一个 tile 内）

**⚠️ 注意：** 对于无法在一行内完成的归约（如超大维度），需要分块归约 + 两遍扫描，这属于高级模式，请先阅读官方参考。

---

### 3. MatMul-like — 矩阵乘类算子

| 子类型      | 示例       | 模板来源                    |
| ----------- | ---------- | --------------------------- |
| 标准 matmul | mm, matmul | `examples/matmul/kernel.py` |
| 批处理      | bmm        | `examples/bmm/kernel.py`    |
| 融合        | addmm      | `examples/addmm/kernel.py`  |

**实现策略：**
- Tile 模式：三级 tiling (M, N, K) + K 维循环
- 核心：`ntl.dot` 在 K 循环中累加
- 累加器：始终用 float32
- `block_size()` 三个维度分别 autotune
- 组合：优先复用 `examples/matmul/kernel.py` 的 `arrangement` 和 `application`

**组合复用模式：**

```python
# bmm: 复用 matmul 的 application，只改 arrangement（加 batch 维）
from examples.matmul.kernel import application

# addmm: 复用 matmul 的 arrangement + application，再加 bias fusion
import examples.matmul.kernel as mm
```

---

### 4. Attention — 注意力算子

| 子类型 | 示例                         | 模板来源                                          |
| ------ | ---------------------------- | ------------------------------------------------- |
| SDPA   | scaled_dot_product_attention | `examples/scaled_dot_product_attention/kernel.py` |
| RoPE   | rotary_position_embedding    | 见下方 P2 新增                                    |

**实现策略：**
- Tile 模式：4D tiling (batch, heads, seq_q, seq_kv)
- 算法：Flash Attention-2 的 online softmax
- 关键技巧：
  - 用 `ntl.exp2` 替代 `ntl.exp`，预乘 `scale * log2(e)` ≈ `scale * 1.44269504089`
  - 维护 running max `m_i` 和 running sum `l_i`
  - KV 边界用 `ntl.where` + `.offsets(-2)` + `.source.shape[-2]`
- `shape_options={"constexpr": True, "upper_bound": 128}` 用于 head_dim

**⚠️ 注意：** Attention 是最复杂的算子族。如果用户请求的 attention 变体与 SDPA 差异较大（如 sparse attention、cross attention with mask），建议先研究官方参考再实现。

---

### 5. Convolution — 卷积算子

| 子类型    | 示例       | 模板来源                          |
| --------- | ---------- | --------------------------------- |
| Conv2D    | conv2d     | 官方参考（im2col + matmul 复用）  |
| MaxPool2D | max_pool2d | 官方参考（window tile + ntl.max） |

**实现策略：**
- Conv2D：**复用 matmul**。将 input 通过 im2col 重排为矩阵，filter flatten 为矩阵，然后调用 matmul。
  - arrangement 中使用 `tile(strides=..., dilation=...)` + `ravel()` + `flatten()` + `permute()`
- MaxPool2D：window-based tiling `(1, 1, WINDOW_H, WINDOW_W)` + `ntl.max(axis=1)`
  - 用 `Tensor(other=float("-inf"))` 处理边界

**⚠️ 注意：** 卷积类的 arrangement 非常复杂（多级 tile + ravel + flatten），强烈建议直接参考官方实现，不要从零推导。

---

### 6. Composition — 组合复用

当用户请求可以通过复用已有算子实现时，优先使用组合模式：

| 目标算子          | 复用来源 | 复用方式                                           |
| ----------------- | -------- | -------------------------------------------------- |
| bmm               | matmul   | 复用 application，自定义 3D arrangement            |
| addmm             | matmul   | 复用 arrangement + application，融合 bias          |
| conv2d            | matmul   | im2col 重排后调用 matmul arrangement + application |
| linear+activation | matmul   | 复用 matmul arrangement，在 application 中追加激活 |

---

### 7. Generator — 生成器算子

| 子类型   | 示例             | 模板来源                                     |
| -------- | ---------------- | -------------------------------------------- |
| 线性生成 | linspace, arange | Pattern 11 in `references/CODE_TEMPLATES.md` |
| 矩阵生成 | eye, ones, zeros | Pattern 11 变体                              |
| 网格生成 | meshgrid         | Pattern 11 + 自定义 arrangement              |

**实现策略：**
- **核心模式：** arrangement 只 tile 输出（无输入 tensor），application 用 `.offsets()` 从索引计算值
- 标量参数（start, end, steps 等）用 `Tensor(0, dtype=float64)` 声明
- 在 wrapper 中预计算步长等值，传入 kernel 避免重复计算
- **block_size 应显式设为 1024**（默认 autotuning 可能选 256，性能差 2x）

**公式速查：**

| 算子     | 公式                                                  |
| -------- | ----------------------------------------------------- |
| linspace | `output[i] = start + i * (end - start) / (steps - 1)` |
| arange   | `output[i] = start + i * step`                        |
| eye      | `output[i,j] = 1.0 if i == j else 0.0`                |

**⚠️ 注意：** 这些算子的 performance bottleneck 是 kernel launch overhead，不是计算量。小 tensor 不可避免比 torch 慢（框架 overhead ~0.027ms vs torch ~0.008ms），大 tensor 可达到 torch 80-90% 的性能。

---

### 8. Scatter — 散布/索引写入算子

| 子类型   | 示例                                   | 模板来源                                        |
| -------- | -------------------------------------- | ----------------------------------------------- |
| 索引写入 | slice_scatter, index_copy, scatter_add | Pattern 13 in `references/CODE_TEMPLATES.md`    |
| 排列写入 | channel_shuffle, scatter               | wrapper-heavy（permute + contiguous）           |
| 视图物化 | moveaxis+contiguous, permute+copy      | Pattern 13 + LIMITATION REPORT + 双路径 wrapper |

**实现策略：**
- **核心模式：** 1D copy kernel（`output_slice = source`）+ wrapper 处理复杂索引
- 九齿的 `output = x` 是全 tile 覆写，**不支持选择性 scatter**
- 跨 tensor gather 索引有 `size_1` 分解限制（详见 `API_REFERENCE.md` §Cross-Tensor Gather Indexing）
- **推荐方案：** wrapper 创建 output slice view → flatten → 传给 1D copy kernel

**关键挑战：**

| 挑战          | 说明                           | 解决方案                                              |
| ------------- | ------------------------------ | ----------------------------------------------------- |
| Strided write | step≠1 导致 output view 非连续 | wrapper fallback 到 PyTorch 切片赋值                  |
| 多维索引      | scatter dim 不是最后一维       | wrapper 中 permute + flatten，或创建 contiguous slice |
| 运行时标量    | start, step, size 等参数       | 使用 Pattern 12（Tensor(0) passthrough）              |
| 负 step       | 部分平台不支持                 | wrapper 中翻转 src + 转换为等价正 step（FC-23）       |

**Wrapper-heavy 模式（适用于 channel_shuffle 等）：**
```python
def channel_shuffle(input, groups):
    B, C, H, W = input.shape
    # 用 PyTorch 的 reshape + transpose + contiguous 实现排列
    # 然后调用 1D copy kernel 将排列后的数据写入 output
    permuted = input.view(B, groups, C // groups, H, W).transpose(1, 2).contiguous().view(B, C, H, W)
    output = torch.empty_like(input)
    kernel(permuted.flatten(), output.flatten())
    return output
```

**⚠️ 注意：** scatter 算子的 kernel 通常非常简单（1D copy），难度在 wrapper 的参数归一化和索引计算。如果发现 kernel 实现过于复杂，考虑是否应该退化为 wrapper-heavy 模式。如果九齿确实无法表达（如 channel_shuffle 的 reshape+transpose 模式），**必须诚实报告**（LIMITATION REPORT），禁止静默用纯 PyTorch 替代。

**视图物化（View Materialization）模式：**

当算子的主操作是视图/元数据操作（moveaxis、permute、narrow 等），虽然主操作无需 kernel，但 `.contiguous()` 路径需要数据拷贝。此时：

1. **主操作**：wrapper 用 `input.permute(perm)` 返回视图（O(1)）
2. **物化路径**：提供 Pattern 13（1D copy kernel）将视图物化为连续 tensor
3. **双路径 wrapper**：默认返回视图，`contiguous=True` 时调用 kernel
4. **LIMITATION REPORT** 说明主操作是视图操作 + kernel 用于物化路径
5. **Benchmark** 必须对比：view-only / kernel contiguous / torch contiguous 三条路径

---

### 9. Layout/View — 布局/视图算子

| 子类型     | 示例                                  | 实现策略                                              |
| ---------- | ------------------------------------- | ----------------------------------------------------- |
| permute 类 | moveaxis, movedim, permute, transpose | 主操作 view-only + Pattern 13 copy kernel（物化路径） |
| 切片类     | narrow, slice, split                  | 主操作 view-only + Pattern 13 copy kernel（物化路径） |
| 重排类     | channel_shuffle, reshape              | wrapper-heavy + Pattern 13 copy kernel                |

**核心认知（CRITICAL）：**

> **视图操作 ≠ 不需要 kernel。** 主操作（`x.permute(perm)`）是 O(1) 的元数据操作，但 `.contiguous()` 路径是 O(N) 的数据拷贝，**需要一个真正的 GPU kernel**。
> PyTorch 的 `x.permute(perm).contiguous()` 底层就是一个 strided copy CUDA kernel。九齿可以写出等价的 kernel。

**实现模式（必须两条路径都提供）：**

```python
# kernels/moveaxis.py — Pattern 13: 1D copy kernel
def arrangement(source, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()
    return source.flatten().tile((block_size,)), output.flatten().tile((block_size,))

def application(source, output):
    output = source  # noqa: F841

# torch/moveaxis.py — 双路径 wrapper
def moveaxis(input, source, destination, *, contiguous=False):
    perm = _compute_moveaxis_perm(input.ndim, source, destination)
    view = input.permute(perm)             # 路径 A: O(1) 视图操作
    if not contiguous:
        return view
    # 路径 B: 九齿 kernel 物化为连续 tensor
    output = torch.empty_like(view)
    kernel = _cached_make(premake, 1, input.dtype)
    kernel(view.flatten(), output.flatten())
    return output
```

**Benchmark 必须对比三条路径：**

| 路径                                               | 预期开销      | 说明                             |
| -------------------------------------------------- | ------------- | -------------------------------- |
| view-only（`x.permute(perm)`）                     | ~2-6 μs       | 纯 Python dispatch               |
| kernel contiguous（九齿 Pattern 13）               | 与 torch 相当 | memory-bound strided copy        |
| torch contiguous（`x.permute(perm).contiguous()`） | 基线          | PyTorch 内部 strided copy kernel |

**⚠️ 常见错误：**
- 只提供了 view-only 路径，没有 kernel → 评审扣分（未使用九齿 DSL）
- 认为"kernel 比 torch.permute 慢所以不需要" → 错误对比（应该比 `.contiguous()` 而不是比 view）
- 写了 LIMITATION REPORT 就停止 → 必须继续提供物化路径的 kernel（见 SKILL.md §Post-Report Action）

---

## 路由规则（给 AI 的指令）

1. 收到请求后，**先在上述分类中定位**算子族。
2. 如果属于「组合复用」表中的目标算子，**直接引用复用路径**，不要从零编写。
3. 如果属于 elementwise 族，**直接套用 1D tile 模板**，注入对应公式。
4. 如果属于 reduction / matmul 族，**引用对应的 examples/ 模板**进行修改。
5. 如果属于 attention / convolution 族，**警告用户这是高级模式**，建议先阅读官方参考和 `references/PATTERNS.md` 中的对应模式。
6. 如果属于 scatter 族，**使用 Pattern 13（1D copy kernel）+ wrapper-heavy 策略**。scatter kernel 保持简单，索引计算放 wrapper。运行时标量用 Pattern 12。详见 FC-22（gather 限制）和 FC-23（平台限制）。
7. 如果属于 layout/view 族，**必须同时提供两条路径**：view-only（O(1) 元数据）+ Pattern 13 copy kernel（`.contiguous()` 物化路径）。不要因为主操作是 view-only 就跳过 kernel 实现。输出 LIMITATION REPORT + 完整 kernel。详见 §9 Layout/View。
8. 如果无法归类，按「自定义算子」处理，参考 `references/OPTIMIZATION_GUIDE.md` 从零设计 tile 策略。
