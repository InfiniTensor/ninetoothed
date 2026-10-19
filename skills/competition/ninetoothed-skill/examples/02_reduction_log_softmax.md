# 示例 2：Reduction 算子开发 — log_softmax

> 本示例展示一个需要数值稳定性的 Reduction 算子如何完成开发。
> 使用 online max + log-sum-exp 技术，一次通过即成功。

## 任务输入

```
CPU 参考实现：
import numpy as np

def log_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
```

## 阶段 1：分析

- **算子类型**：Reduction（模式 2）
- **输入**：N 维 tensor
- **输出**：N 维 tensor（与输入同 shape/dtype）
- **特殊参数**：`dim`（规约维度）
- **数值挑战**：
  - 直接 `log(sum(exp(x)))` 会溢出
  - float16 中间计算需要提升精度
- **共享 arrangement**：`kernels/reduction.py`

## 阶段 2：生成

```python
# kernels/log_softmax.py
import functools
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ntops.kernels.reduction import arrangement


def application(input, output):
    dtype = input.dtype.dtype
    exp_dtype = dtype if dtype != ntl.float16 else ntl.float32

    # Pass 1: online max + exp 累加
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

    # Pass 2: 计算 x - max - log(sum_exp)
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

## 关键技术点

### Online Softmax 算法

传统 softmax/log_softmax 需要两遍扫描（一遍找 max，一遍计算），online softmax 在一遍中完成：

1. 维护 `prev_max`（当前最大值）和 `accumulator`（累加和）
2. 每遇到新的 block，更新 `curr_max`
3. 用 `prev_curr_diff_exp` 修正之前的累加值：`acc = acc * exp(prev_max - curr_max) + sum(exp(x - curr_max))`
4. 最终 `accumulator` 就是 `sum(exp(x - max))`

### float16 精度提升

```python
exp_dtype = dtype if dtype != ntl.float16 else ntl.float32
```

float16 的 exp 和 log 运算范围有限（exp 最大约 11），必须提升到 float32。

### input padding

```python
Tensor(ndim, dtype=dtype, other=float("-inf"), shape_options={"constexpr": True})
```

- `other=float("-inf")`：padding 区域的 exp 值为 0，不影响 sum
- `shape_options={"constexpr": True}`：让 shape 相关值在编译时确定

## 阶段 4：精度验证

```
test_log_softmax_float32_basic PASSED
test_log_softmax_float16_basic PASSED
test_log_softmax_dim0 PASSED
test_log_softmax_3d PASSED
test_log_softmax_numerical_stability PASSED
test_log_softmax_edge_cases PASSED
```

数值稳定性测试使用极端输入值（-10000 到 10000），验证无 NaN/Inf。

## 阶段 5：性能评估

| 算子 | 输入规模 | ntops | PyTorch | 比率 |
|------|----------|-------|---------|------|
| log_softmax | 4096x1024 | 0.197ms | 0.160ms | 0.81x |

六项策略评估：
1. ✅ 内存访问：reduction arrangement 保证沿规约维度 coalesced access
2. ⬜ 算子融合：log 本身已融合 max + sum + log，无进一步融合空间
3. ⬜ 循环展开：range 循环由 triton 自动展开
4. ⬜ 减少同步：两次 pass 之间无额外同步
5. ✅ 精度策略：float16 中间计算提升到 float32
6. ✅ 计算重组：online softmax 避免二次遍历

## 关键经验

**Reduction 算子数值稳定性**：
- 始终使用 online max-normalization 技术避免 exp 溢出
- float16 中间计算必须提升精度
- padding 值设为 `float("-inf")` 确保 padding 区域不影响累加
