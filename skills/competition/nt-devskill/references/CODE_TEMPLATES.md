# NineToothed Arrangement 代码模板库

> 每个模板可直接复制使用。

---

## Pattern 1: 1D Elementwise（逐元素）

适用于：add, mul, relu, silu, gelu, abs, neg, exp 等逐元素操作。

```python
import functools
from ninetoothed import Symbol, Tensor
import ninetoothed.language as ntl

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    input_arranged = input.tile((BLOCK_SIZE,))
    output_arranged = output.tile((BLOCK_SIZE,))
    return input_arranged, output_arranged

def application(input, output):
    output = input + 1  # 替换为实际计算
```

**Tensor 声明：** `tensors = (Tensor(1), Tensor(1))`

---

## Pattern 2: 1D Binary（双输入逐元素）

适用于：add, mul, copysign, maximum, minimum, pow 等双输入操作。

```python
def arrangement(a, b, output, BLOCK_SIZE=BLOCK_SIZE):
    a_arr = a.tile((BLOCK_SIZE,))
    b_arr = b.tile((BLOCK_SIZE,))
    out_arr = output.tile((BLOCK_SIZE,))
    return a_arr, b_arr, out_arr

def application(a, b, output):
    output = a + b  # 替换为实际计算
```

**Tensor 声明：** `tensors = (Tensor(1), Tensor(1), Tensor(1))`

---

## Pattern 3: 1D with Scalar Parameter（带标量参数）

适用于：leaky_relu(slope), clamp(min, max), scale(alpha) 等带标量参数的操作。

```python
def arrangement(input, output, slope, BLOCK_SIZE=BLOCK_SIZE):
    input_arr = input.tile((BLOCK_SIZE,))
    output_arr = output.tile((BLOCK_SIZE,))
    return input_arr, output_arr, slope

def application(input, output, slope):
    output = ntl.where(input >= 0, input, slope * input)
```

**Tensor 声明：** `tensors = (Tensor(1), Tensor(1), Tensor(0))` — `Tensor(0)` 表示标量

---

## Pattern 4: 2D Row Reduction（行归约）

适用于：softmax, sum, mean, max, min, norm 等沿最后一维归约。

```python
def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    input_arranged = input.tile((1, -1))   # 每次取一行
    output_arranged = output.tile((1,))    # 输出标量 per row
    return input_arranged, output_arranged

def application(input, output):
    output = ntl.sum(input, axis=1)  # 替换为实际归约
```

**Tensor 声明：** `tensors = (Tensor(2), Tensor(1))`

> **注意：** BLOCK_SIZE 应 ≥ input.shape[-1]，确保一行完全装入一个 tile。

---

## Pattern 5: 2D Row-wise with Full Row（整行 tile）

适用于：softmax, rms_norm 等需要整行数据的归约操作。

```python
def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    # tile 成 (1, BLOCK_SIZE)，BLOCK_SIZE = input.shape[-1]
    input_arranged = input.tile((1, BLOCK_SIZE))
    output_arranged = output.tile((1, BLOCK_SIZE))
    return input_arranged, output_arranged

def application(input, output):
    x = ntl.cast(input, ntl.float32)
    # ... 计算逻辑 ...
    output = result.to(output.dtype)
```

**调用时：** `BLOCK_SIZE=input.shape[-1]`

---

## Pattern 6: 3D MatMul（矩阵乘法 + K-loop）

适用于：matmul, bmm, addmm 等矩阵乘法操作。

```python
BLOCK_M = Symbol("BLOCK_M", constexpr=True)
BLOCK_N = Symbol("BLOCK_N", constexpr=True)
BLOCK_K = Symbol("BLOCK_K", constexpr=True)

def arrangement(a, b, output, BM=BLOCK_M, BN=BLOCK_N, BK=BLOCK_K):
    out_arr = output.tile((BM, BN))

    a_arr = a.tile((BM, BK)).tile((1, -1)).expand((-1, out_arr.shape[1]))
    a_arr.dtype = a_arr.dtype.squeeze(0)

    b_arr = b.tile((BK, BN)).tile((-1, 1)).expand((out_arr.shape[0], -1))
    b_arr.dtype = b_arr.dtype.squeeze(1)

    return a_arr, b_arr, out_arr

def application(a, b, output):
    acc = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(a.shape[0]):
        acc += ntl.dot(a[k], b[k])
    output = acc.to(output.dtype)
```

**Tensor 声明：** `tensors = (Tensor(2), Tensor(2), Tensor(2))`

---

## Pattern 7: Batched 3D（批量操作）

适用于：bmm, batched_add 等需要批处理维度的操作。复用 Pattern 6 的 application。

```python
def arrangement(a, b, output, BM=BLOCK_M, BN=BLOCK_N, BK=BLOCK_K):
    out_arr = output.tile((1, BM, BN))  # 批次维度不 tile

    a_arr = a.tile((1, BM, BK)).tile((1, 1, -1)).expand((1, -1, out_arr.shape[2]))
    a_arr.dtype = a_arr.dtype.dtype.squeeze(0)

    b_arr = b.tile((1, BK, BN)).tile((1, -1, 1)).expand((1, out_arr.shape[1], -1))
    b_arr.dtype = b_arr.dtype.dtype.squeeze(1)

    return a_arr, b_arr, out_arr
```

**Tensor 声明：** `tensors = (Tensor(3), Tensor(3), Tensor(3))`

---

## Pattern 8: 4D Attention（注意力机制）

适用于：scaled_dot_product_attention, multi-head attention。

```python
def arrangement(q, k, v, output, BM=BLOCK_M, BN=BLOCK_N, BK=BLOCK_K):
    q_arr = q.tile((1, 1, BM, BK))
    k_arr = k.tile((1, 1, BN, BK)).tile((1, 1, -1, 1))
    k_arr.expand((1, 1, -1, q_arr.shape[2], -1))
    k_arr.dtype = k_arr.dtype.dtype.squeeze(1)
    # ... 类似处理 v 和 output
    return q_arr, k_arr, v_arr, out_arr
```

> **注意：** 4D tiling 需要 `dtype.dtype.squeeze()` 双层访问。

---

## Pattern 9: Strided/Windowed Tiling（步幅/窗口 tiling）

适用于：conv2d (im2col), max_pool2d, unpool。

```python
def arrangement(input, output, ...):
    input_arr = input.tile((1, 1, KERNEL_H, KERNEL_W),
                           strides=(1, 1, stride_h, stride_w),
                           dilation=(1, 1, 1, 1))
    # 用 ravel() + flatten() 展平窗口维度
    input_arr = input_arr.ravel()
    input_arr = input_arr.flatten(end_dim=2, start_dim=1)
    # ... permute 后送入 matmul
```

---

## Pattern 10: Flatten-then-Kernel（展平后 kernel）

适用于：无法直接 tiling 的复杂操作，先 reshape 到 1D 再用 1D kernel。

```python
# 在 wrapper 中:
flat_input = input.reshape(-1)
flat_output = torch.empty_like(flat_input)
kernel(flat_input, flat_output)
output = flat_output.reshape(input.shape)
```

---

## Pattern 11: Generator / Index-based（从索引生成值）

适用于：linspace, arange, eye, meshgrid, ones, zeros 等**不读取输入 tensor，而是根据元素索引生成输出值**的算子。

**核心区别：** 这些算子没有输入 tensor（或只有标量参数），arrangement 只 tile 输出，application 用 `.offsets()` 获取元素索引并计算值。

```python
import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, Symbol

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

# arrangement: 只 tile 输出（没有输入 tensor）
def arrangement(output, start, step, BLOCK_SIZE=BLOCK_SIZE):
    return output.tile((BLOCK_SIZE,)), start, step

# application: 用 .offsets(0) 获取元素索引，从索引计算值
def application(output, start, step):
    indices = ntl.cast(output.offsets(0), ntl.float32)
    output = start + indices * step  # noqa: F841

# premake: Tensor(0) 用于标量参数
def premake(dtype=None, block_size=1024):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (
        Tensor(1, dtype=dtype),                # output
        Tensor(0, dtype=ninetoothed.float64),  # start (scalar)
        Tensor(0, dtype=ninetoothed.float64),  # step (scalar)
    )
    return arrangement_, application, tensors
```

**关键点：**
- `Tensor(0)` 声明标量参数（如 start, end, steps），不参与 tile
- `.offsets(0)` 返回当前 tile 中每个元素的全局索引（1D 数组）
- 在 wrapper 中预计算步长等值，传给 kernel 避免重复计算
- **block_size 应显式设为 1024**（默认 autotuning 可能选 256，性能差 2x）

**Wrapper 模式：**
```python
def linspace(start, end, steps, *, dtype=None, device=None, out=None):
    if dtype is None: dtype = torch.float32
    if device is None: device = "cuda"
    if out is None: out = torch.empty(steps, dtype=dtype, device=device)

    step = 0.0 if steps == 1 else (end - start) / (steps - 1)
    kernel = _cached_make(ntops.kernels.linspace.premake)
    kernel(out, float(start), float(step))
    return out
```

**测试要点：**
- 多种参数组合（正/负范围、steps=1 边界、反向 range）
- 多种 dtype（float32, float16）
- 边界精度检查（首尾元素必须精确匹配 start 和 end）

---

## Pattern 12: Runtime Scalar Parameter Passing（运行时标量传参）

适用于：需要将运行时计算的整数参数（如 start, step, size 等）传给 kernel 的算子，如 slice_scatter, narrow_with_offset, indexed_copy 等。

**核心区别：** 与 Pattern 3（constexpr Symbol 标量）不同，这里的标量值在**每次调用时可能不同**，不能作为编译期常量。必须用 `Tensor(0)` 声明，并在 arrangement 中 **passthrough**（原样传出）。

```python
import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

# arrangement: Tensor(0) 标量必须在返回元组中 passthrough
def arrangement(input, start, step, src_size, output, BLOCK_SIZE=BLOCK_SIZE):
    return (
        input.flatten().tile((BLOCK_SIZE,)),
        start,           # ← passthrough: 原样传入，原样传出
        step,            # ← passthrough
        src_size,        # ← passthrough
        output.flatten().tile((BLOCK_SIZE,)),
    )

# application: 标量可直接参与运算
def application(input, start, step, src_size, output):
    offsets = input.offsets(0)
    target_pos = start + offsets * step
    valid = (offsets < src_size) & (target_pos >= 0)
    # ... 计算逻辑 ...

# premake: Tensor(0, dtype=int32) 声明每个标量
def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, BLOCK_SIZE=block_size)
    tensors = (
        Tensor(ndim, dtype=dtype),             # input
        Tensor(0, dtype=ninetoothed.int32),    # start (scalar)
        Tensor(0, dtype=ninetoothed.int32),    # step (scalar)
        Tensor(0, dtype=ninetoothed.int32),    # src_size (scalar)
        Tensor(ndim, dtype=dtype),             # output
    )
    return arrangement_, application, tensors
```

**关键点（CRITICAL）：**

1. **必须 passthrough**：`Tensor(0)` 标量必须出现在 arrangement 的返回元组中。如果只在参数中接收但不返回，生成的 launcher 函数不会包含这些参数 → 调用时报 `TypeError: takes N positional arguments but M were given`。
2. **调用时传 Python int**：wrapper 中直接传 `int(value)` 即可，ninetoothed 自动转为 Triton 标量。
3. **标量位置必须匹配**：arrangement 返回元组中的标量位置 = kernel 调用时的参数位置 = tensors 元组中的位置。三者必须一一对应。
4. **不要在 arrangement 中对 Tensor(0) 做 Python 运算**：如 `scalar_a + scalar_b` 会报 `TypeError: unsupported operand type(s) for +: 'Tensor' and 'Tensor'`。

**Wrapper 模式：**
```python
def my_op(input, start, step, actual_len):
    # parameter normalization, clamping, etc. done here
    kernel = _cached_make(ntops.kernels.my_op.premake, input.ndim, input.dtype)
    kernel(input, int(start), int(step), int(actual_len), output)
    return output
```

---

## Pattern 13: 1D Copy Kernel（一维拷贝 / scatter-write）

适用于：slice_scatter, index_copy, scatter 等需要将 source 数据写入 output 特定位置的算子。当跨 tensor 索引（gather）受限于九齿的 `size_1` 分解问题时，退化为纯 1D copy + wrapper 处理复杂索引。

**核心思路：** kernel 只做 `output[i] = source[i]` 的简单拷贝，所有 scatter 索引计算在 wrapper 中完成。

```python
import functools
import ninetoothed
from ninetoothed import Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(source, output_slice, BLOCK_SIZE=BLOCK_SIZE):
    return (
        source.flatten().tile((BLOCK_SIZE,)),
        output_slice.flatten().tile((BLOCK_SIZE,)),
    )

def application(source, output_slice):
    output_slice = source  # noqa: F841

def premake(ndim=1, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, BLOCK_SIZE=block_size)
    tensors = (
        Tensor(ndim, dtype=dtype),   # source (1D flattened)
        Tensor(ndim, dtype=dtype),   # output_slice (1D, view of output)
    )
    return arrangement_, application, tensors
```

**Wrapper 模式（处理复杂索引）：**
```python
def slice_scatter(input, src, dim=0, start=0, end=None, step=1):
    # 1. 参数归一化（负索引、越界截断等）
    # 2. output = input.clone()
    # 3. 创建 output 的 slice view
    slices = [slice(None)] * input.ndim
    slices[dim] = slice(start, end, step)
    output_view = output[tuple(slices)]

    if output_view.is_contiguous():
        # contiguous view → 用九齿 1D copy kernel
        kernel = _cached_make(premake, 1, input.dtype)
        kernel(src.flatten(), output_view.flatten())
    else:
        # non-contiguous view (step≠1) → fallback to torch slice assignment
        output[tuple(slices)] = src

    return output
```

**关键点：**
- 九齿的 `output = source` 是**全 tile 覆写**，不支持选择性 scatter
- 跨 tensor 索引 `tensor[idx]` 有 `size_1` 分解限制（详见 API_REFERENCE.md）
- 复杂 scatter 的索引计算放在 wrapper 中，kernel 保持最简

---

## Pattern 14: Broadcast Elementwise（广播逐元素 + mask/dtype 约束）

适用于：输入 tensor shape 不同需要广播的逐元素算子，或带 mask/条件约束的算子（masked_fill, where, clamp 等）。

**14A. 广播输入（不同 shape）：**

```python
# wrapper 中先 broadcast 到相同 shape，再传给 kernel
def my_op(a, b):
    # torch.broadcast_tensors 处理所有广播规则
    a, b = torch.broadcast_tensors(a, b)
    output = torch.empty_like(a)
    kernel(a.flatten(), b.flatten(), output.flatten())
    return output

# kernel 用标准 1D tile（broadcast 后 shape 一致）
def arrangement(a, b, output, BLOCK_SIZE=BLOCK_SIZE):
    return a.tile((BLOCK_SIZE,)), b.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

def application(a, b, output):
    output = a + b  # 替换为实际计算
```

**关键点：** 在 wrapper 中用 `torch.broadcast_tensors` 展开到相同 shape 后再传给 kernel。kernel 内部不需要处理广播逻辑。

**14B. Mask 约束（条件计算作为主操作）：**

适用于 masked_fill、where（作为算子本身）、clamp 等以条件判断为核心计算的算子。

```python
def application(input, mask, output):
    # mask 是 bool tensor，与 input 同 shape
    output = ntl.where(mask, input, ntl.cast(0.0, input.dtype))  # noqa: F841
    # 或：masked_fill 语义
    # output = ntl.where(mask, fill_value, input)  # noqa: F841
```

**14C. Dtype 约束（dtype 依赖的计算路径）：**

当算子对不同 dtype 有不同计算路径时（如 copysign 对 fp16/fp32/fp64 用不同位宽操作）：

```python
def application(input, output):
    if input.dtype is ntl.float16:
        # fp16 专用路径（位操作）
        bits = ntl.cast(input, ntl.uint16, bitcast=True)
        result = bits & ntl.cast(0x7FFF, ntl.uint16)
        output = ntl.cast(result, ntl.float16, bitcast=True)  # noqa: F841
    elif input.dtype is ntl.float32:
        # fp32 路径
        x_f32 = ntl.cast(input, ntl.float32)
        output = computation(x_f32).to(input.dtype)  # noqa: F841
```

**关键点：**
- `input.dtype is ntl.float16` 在编译期求值（九齿编译器按 dtype 分支生成不同 kernel）
- 所有中间计算用 fp32，最终 `.to(input.dtype)` 写回原始精度
- 在 wrapper 中不需要特殊处理 dtype——kernel 自动适配

---

## Pattern 15: Chunked / Multi-pass Reduction（分块归约）

适用于：归约维度超过单个 tile 容量（如 softmax 的行长度 > BLOCK_SIZE），或需要"softmax 子任务"（只算 max、只算 sum、log-sum-exp 等）。

**15A. 两遍扫描归约（Two-pass Reduction）：**

当行长度超过 tile 容量时，用两遍扫描：第一遍算 max，第二遍算 sum。

```python
# === Kernel 1: 分块 max ===
def arrangement_max(input, partial_max, BLOCK_SIZE=BLOCK_SIZE):
    # input: tile((1, BLOCK_SIZE)) — 每次取一段
    # partial_max: tile((1,)) — 每段输出一个标量
    return input.tile((1, BLOCK_SIZE)), partial_max.tile((1,))

def application_max(input, partial_max):
    x = ntl.cast(input, ntl.float32)
    partial_max = ntl.max(x, axis=1)  # noqa: F841

# === Kernel 2: 用全局 max 计算 exp-sum ===
def arrangement_sum(input, global_max, partial_sum, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((1, BLOCK_SIZE)), global_max, partial_sum.tile((1,))

def application_sum(input, global_max, partial_sum):
    x = ntl.cast(input, ntl.float32)
    shifted = x - global_max[:, None]  # broadcast 全局 max
    partial_sum = ntl.sum(ntl.exp(shifted), axis=1)  # noqa: F841
```

**Wrapper 编排：**
```python
def chunked_softmax(input):
    rows, cols = input.shape
    BLOCK = 2048  # tile 容量
    n_chunks = (cols + BLOCK - 1) // BLOCK

    # Pass 1: 分块 max → 取全局 max
    partial_max = torch.empty(rows, n_chunks, device=input.device)
    kernel_max(input, partial_max)
    global_max = partial_max.max(dim=1).values  # 跨 chunk 取 max

    # Pass 2: 用全局 max 计算 exp-sum
    partial_sum = torch.empty(rows, n_chunks, device=input.device)
    kernel_sum(input, global_max, partial_sum)
    global_sum = partial_sum.sum(dim=1)  # 跨 chunk 求和

    return global_max, global_sum
```

**15B. Softmax 子任务（只算部分）：**

| 子任务       | 公式                           | 用哪个 kernel                   |
| ------------ | ------------------------------ | ------------------------------- |
| row max      | `max(x, axis=1)`               | Kernel 1 + 跨 chunk max         |
| row sum      | `sum(x, axis=1)`               | 单遍 kernel（无需 max）         |
| log-sum-exp  | `log(sum(exp(x - max))) + max` | Kernel 1 + Kernel 2 + log       |
| 完整 softmax | `exp(x - max) / sum`           | 标准 Pattern 4/5（行能 fit 时） |

**15C. Online Softmax（单遍，用于 attention）：**

当在 K 循环中逐 tile 累积时，用 online softmax 避免两遍扫描：

```python
# 在 attention 的 K-loop 中
m_i = ntl.full((M,), float("-inf"), ntl.float32)  # running max
l_i = ntl.zeros((M,), ntl.float32)                 # running sum

for k in range(num_k_tiles):
    qk = ntl.dot(q, k_tiles[k])
    m_new = ntl.maximum(m_i, ntl.max(qk, axis=1))
    alpha = ntl.exp2((qk - m_new[:, None]) * scale_log2e)
    l_i = l_i * ntl.exp2((m_i - m_new) * scale_log2e) + ntl.sum(alpha, axis=1)
    m_i = m_new
```

**关键点（CRITICAL）：**
- BLOCK_SIZE 必须 ≤ 行长度，否则 partial tile 的 max/sum 不正确
- 跨 chunk 的 max 用 `torch.max`（在 wrapper 中），不要在 kernel 间传递
- fp32 累加器是必须的（FC-15）
- 对于 attention，online softmax 比两遍扫描更高效（只需一遍 K-loop）

---

## 选择指南

| 算子特征                        | 推荐 Pattern    |
| ------------------------------- | --------------- |
| 1:1 输入输出映射                | Pattern 1 or 2  |
| 带编译期标量参数                | Pattern 3       |
| 带运行时标量参数                | **Pattern 12**  |
| 广播/不同 shape 输入            | **Pattern 14A** |
| mask/条件约束算子               | **Pattern 14B** |
| dtype 依赖路径                  | **Pattern 14C** |
| 沿轴归约（行能 fit 在 tile 内） | Pattern 4 or 5  |
| 沿轴归约（行超过 tile 容量）    | **Pattern 15A** |
| softmax 子任务（max/sum/lse）   | **Pattern 15B** |
| attention 中的 online softmax   | **Pattern 15C** |
| 矩阵乘法                        | Pattern 6       |
| 批量矩阵乘法                    | Pattern 7       |
| 注意力机制                      | Pattern 8       |
| 卷积/池化                       | Pattern 9       |
| 复杂 reshape                    | Pattern 10      |
| 从索引生成值（无输入 tensor）   | Pattern 11      |
| scatter/index_copy 类写入       | **Pattern 13**  |
