# 九种 Arrangement 模式代码模板

每个模板包含 kernel 文件（`kernels/{op_name}.py`）和 torch 文件（`torch/{op_name}.py`）的完整骨架。

---

## 模板 1：Element-wise Unary（relu, sin, exp, neg, abs, silu, sigmoid...）

### Kernel 文件

```python
import functools
import ninetoothed.language as ntl  # 按需
from ninetoothed import Tensor
from ntops.kernels.element_wise import arrangement


def application(input, output):
    output = {COMPUTATION}(input)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
    return arrangement_, application, tensors
```

### Torch 文件

```python
import torch
import ntops
from ntops.torch.utils import _cached_make


def {op_name}(input):
    output = torch.empty_like(input)
    kernel = _cached_make(ntops.kernels.{op_name}.premake, input.ndim)
    kernel(input, output)
    return output
```

### COMPUTATION 映射表

| CPU 实现 | NineToothed application |
|----------|------------------------|
| `max(0, x)` | `max(0.0, input)` |
| `x / (1 + exp(-x))` | `input / (1 + ntl.exp(-ntl.cast(input, ntl.float32)))` |
| `1 / (1 + exp(-x))` | `1 / (1 + ntl.exp(-ntl.cast(input, ntl.float32)))` |
| `sin(x)` | `ntl.sin(ntl.cast(input, ntl.float32))` |
| `exp(x)` | `ntl.exp(ntl.cast(input, ntl.float32))` |
| `-x` | `-input` |
| `abs(x)` | `ntl.abs(input)` |
| `tanh(x)` | `ntl.tanh(ntl.cast(input, ntl.float32))` |
| `1 / sqrt(x)` | `1 / ntl.sqrt(ntl.cast(input, ntl.float32))` |

---

## 模板 2：Element-wise Binary（add, mul, sub, div, eq, lt, gt...）

### Kernel 文件

```python
import functools
from ninetoothed import Tensor
from ntops.kernels.element_wise import arrangement


def application(input, other, output):
    output = input {OP} other  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
    return arrangement_, application, tensors
```

### OP 映射表

| CPU 实现 | OP |
|----------|-----|
| `a + b` | `+` |
| `a * b` | `*` |
| `a - b` | `-` |
| `a / b` | `/` |
| `a == b` | 须用 `ntl.where(input == other, 1, 0)` |
| `a < b` | 须用 `ntl.where(input < other, 1, 0)` |
| `a & b` | `&` |
| `a \| b` | `\|` |

---

## 模板 3：Element-wise with Scalar

### 方式 A：标量作为同类型 Tensor（clamp 模式）

```python
# kernels/clamp.py
import functools
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ntops.kernels.element_wise import arrangement


def application(input, min_val, max_val, output):
    result = ntl.minimum(input, max_val)
    output = ntl.maximum(result, min_val)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),  # min_val
        Tensor(ndim, dtype=dtype),  # max_val
        Tensor(ndim, dtype=dtype),
    )
    return arrangement_, application, tensors
```

### 方式 B：标量作为 constexpr（leaky_relu 模式）

```python
# kernels/leaky_relu.py
import functools
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ntops.kernels.element_wise import arrangement


def application(input, output, negative_slope):
    output = ntl.where(input >= 0, input, negative_slope * input)  # noqa: F841


def premake(ndim, negative_slope=0.01, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(0, constexpr=True, value=negative_slope),
    )
    return arrangement_, application, tensors
```

### 方式 A Torch 文件（clamp）

```python
# torch/clamp.py
import torch
import ntops
from ntops.torch.utils import _cached_make


def clamp(input, min=None, max=None, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    kernel = _cached_make(ntops.kernels.clamp.premake, input.ndim)
    kernel(input, min, max, out)
    return out
```

### 方式 B Torch 文件（leaky_relu）

```python
# torch/leaky_relu.py
import torch
import ntops
from ntops.torch.utils import _cached_make


def leaky_relu(input, negative_slope=0.01, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    kernel = _cached_make(ntops.kernels.leaky_relu.premake, input.ndim, negative_slope)
    kernel(input, output, negative_slope)  # constexpr 仍需在运行时传入
    return output
```

### 选择规则

- 标量与输入逐元素运算（clamp）→ 方式 A
- 标量作为条件系数或编译时确定（slope, precision）→ 方式 B
- 整数枚举常量 → 方式 B

---

## 模板 4：Reduction（softmax, rms_norm, layer_norm, log_softmax）

### Softmax

```python
import functools
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ntops.kernels.reduction import arrangement


def application(input, output):
    max_val = ntl.full(input.dtype.shape, float("-inf"), dtype=ntl.float32)
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        max_val = ntl.maximum(max_val, input_i)

    accumulator = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    total = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        exp_val = ntl.exp(input_i - max_val)
        accumulator += exp_val * input_i
        total += exp_val

    output = accumulator / total  # noqa: F841


def premake(ndim, dim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)
    tensors = (
        Tensor(ndim, dtype=dtype, other=float("-inf")),
        Tensor(ndim, dtype=dtype),
    )
    return arrangement_, application, tensors
```

### Reduction 关键规则

- 使用 `input.dtype.shape` 获取内层 block 的 shape
- 累加器始终用 float32
- 输入需要 `ntl.cast(input[i], ntl.float32)`
- softmax 输入设 `other=float("-inf")`，norm 类设 `other=0`
- 用 `range(input.shape[0])` 遍历外层 tile 维度

### 模板 4b：log_softmax（reduction + 数值稳定 log-sum-exp）

log_softmax 复用 softmax 的 online max → exp 累加模式，最终输出 `x - max - log(sum_exp)`：

```python
# kernels/log_softmax.py
import functools
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ntops.kernels.reduction import arrangement


def application(input, output):
    dtype = input.dtype.dtype
    exp_dtype = dtype if dtype != ntl.float16 else ntl.float32

    prev_max = ntl.cast(float("-inf"), dtype)
    accumulator = ntl.cast(0, dtype)

    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], dtype)
        curr_max = ntl.cast(ntl.maximum(prev_max, ntl.max(input_i)), dtype)
        prev_curr_diff_exp = ntl.cast(
            ntl.exp(ntl.cast(prev_max - curr_max, exp_dtype)), dtype
        )
        input_curr_diff_exp = ntl.cast(
            ntl.exp(ntl.cast(input_i - curr_max, exp_dtype)), dtype
        )
        accumulator = accumulator * prev_curr_diff_exp + ntl.sum(input_curr_diff_exp)
        prev_max = curr_max

    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], dtype)
        output[i] = ntl.cast(
            ntl.cast(input_i, exp_dtype) - ntl.cast(prev_max, exp_dtype)
            - ntl.cast(ntl.log(ntl.cast(accumulator, exp_dtype)), exp_dtype),
            dtype,
        )  # noqa: F841


def premake(ndim, dim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)
    tensors = (
        Tensor(ndim, dtype=dtype, other=float("-inf"), shape_options={"constexpr": True}),
        Tensor(ndim, dtype=dtype),
    )
    return arrangement_, application, tensors
```

**log_softmax 要点：**
- 与 softmax 共享第一个 pass（online max + exp 累加），但不存储中间 softmax 值
- 第二个 pass 直接计算 `log(exp(x-max) / sum) = x - max - log(sum)`
- float16 中间计算必须提升到 float32（exp 和 log 都会溢出）
- 使用 `ntl.log` 取对数，输入需 cast 到 float32 防止精度丢失

```python
# torch/log_softmax.py
import torch
import ntops
from ntops.torch.utils import _cached_make


def log_softmax(input, dim, dtype=None):
    tensor_dtype = dtype if dtype is not None else input.dtype
    output = torch.empty_like(input, dtype=tensor_dtype)
    kernel = _cached_make(ntops.kernels.log_softmax.premake, input.ndim, dim)
    kernel(input, output)
    return output
```

---

## 模板 5：Matmul（mm）

```python
import enum
import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE_M = ninetoothed.block_size()
BLOCK_SIZE_N = ninetoothed.block_size()
BLOCK_SIZE_K = ninetoothed.block_size()


class InputPrecisionVariant(enum.IntEnum):
    TF32 = enum.auto()
    IEEE = enum.auto()


def arrangement(input, other, output, input_precision,
                block_size_m=None, block_size_n=None, block_size_k=None):
    if block_size_m is None: block_size_m = BLOCK_SIZE_M
    if block_size_n is None: block_size_n = BLOCK_SIZE_N
    if block_size_k is None: block_size_k = BLOCK_SIZE_K

    output_arranged = output.tile((block_size_m, block_size_n))
    input_arranged = input.tile((block_size_m, block_size_k))
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    other_arranged = other.tile((block_size_k, block_size_n))
    other_arranged = other_arranged.tile((-1, 1))
    other_arranged = other_arranged.expand((output_arranged.shape[0], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze(1)

    return input_arranged, other_arranged, output_arranged, input_precision


def application(input, other, output, input_precision):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    if input_precision == 2:
        input_precision_: ntl.constexpr = "ieee"
    else:
        input_precision_: ntl.constexpr = "tf32"
    for k in range(input.shape[0]):
        accumulator += ntl.dot(input[k], other[k], input_precision=input_precision_)
    output = accumulator  # noqa: F841


def premake(input_precision=None, dtype=None, block_size_m=None,
            block_size_n=None, block_size_k=None):
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )
    tensors = (
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=dtype),
        Tensor(0, constexpr=True, value=input_precision),
    )
    return arrangement_, application, tensors
```

### Matmul 关键规则

- 三个独立的 block_size（M, N, K），全部用 `block_size()` 自动调优
- tile-expand-squeeze 模式：tile 分 K → expand 广播 → squeeze 去退化维
- input_precision 用 `Tensor(0, constexpr=True, value=...)` 编译时传入
- 用 `ntl.dot()` 做块内矩阵乘

---

## 模板 6：Composed（addmm, conv2d 复用 mm）

```python
import copy
import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ntops.kernels import mm


def arrangement(input, x, y, output, input_precision,
                block_size_m=None, block_size_n=None, block_size_k=None):
    _, _, input_arranged, _ = mm.arrangement(
        x, y, input, input_precision,
        block_size_m=block_size_m, block_size_n=block_size_n,
        block_size_k=block_size_k,
    )
    x_arranged, y_arranged, output_arranged, _ = mm.arrangement(
        copy.deepcopy(x), copy.deepcopy(y), output, input_precision,
        block_size_m=block_size_m, block_size_n=block_size_n,
        block_size_k=block_size_k,
    )
    return input_arranged, x_arranged, y_arranged, output_arranged, input_precision


def application(input, x, y, beta, alpha, output, input_precision):
    mm_output = ntl.zeros(output.shape, dtype=ntl.float32)
    mm.application(x, y, mm_output, input_precision)
    output = ntl.cast(beta * ntl.cast(input, ntl.float32)
                       + alpha * mm_output, output.dtype.dtype)  # noqa: F841
```

### Composed 关键规则

- 多次调用 mm.arrangement 时必须用 `copy.deepcopy` 复制 tensor 模板
- application 内可调用 `mm.application` 复用计算逻辑
- 结果需要 `ntl.cast` 回原始 dtype

---

## 模板 7：Attention（Flash Attention）

4D tensor，支持 GQA、kv cache、causal masking。

### 关键特征

- 4D tensor: `[batch, heads, seq, dim]`
- `shape_options` 约束 head 维度：`{"constexpr": True, "upper_bound": 128}`
- online softmax：运行 max → exp2 trick → 累加 → 归一化
- GQA 支持：`query.shape[-3] // key.shape[-3]` 计算比例
- 多个 constexpr 标量：`is_causal`, `with_attn_mask`, `causal_variant`
- `ntl.offsets(-2)` 用于位置信息，`ntl.where` 用于 causal mask

---

## 模板 8：Pooling（max_pool2d, avg_pool2d）

```python
# max_pool2d application
def application(input, output):
    output = ntl.max(input, axis=-1)  # noqa: F841

# avg_pool2d application
def application(input, output):
    output = ntl.sum(input, axis=-1) / input.shape[-1]  # noqa: F841
```

使用 `kernels/pooling.py` 的共享 arrangement。输入设 `other=float("-inf")`（max）或 `other=0`（avg）。

---

## 模板 9：RoPE（Rotary Position Embedding）

自定义 arrangement 支持交错/非交错模式：
- 交错：dilation `(1,1,1,2)` 在最后一维做 stride-2 采样
- 非交错：标准 tile
- Application 拆分输入对，应用旋转：`x0 * cos - x1 * sin`, `x0 * sin + x1 * cos`
