# 示例 1：Element-wise 算子开发 — leaky_relu

> 本示例展示一个带标量参数的 Element-wise 算子如何从 CPU 参考实现完成完整开发闭环。
> 经历了 5 次迭代修复，最终发现 constexpr 是传递标量系数的正确方式。

## 任务输入

```
CPU 参考实现：
import numpy as np

def leaky_relu(x, negative_slope=0.01):
    return np.where(x >= 0, x, negative_slope * x)
```

## 阶段 1：分析

- **算子类型**：Element-wise（模式 1）
- **输入**：1 个 N 维 tensor
- **输出**：1 个 N 维 tensor（与输入同 shape/dtype）
- **特殊参数**：`negative_slope`（标量，默认 0.01）
- **边界情况**：negative_slope 可以是任意正浮点数
- **共享 arrangement**：`kernels/element_wise.py`

## 阶段 2：生成 — 迭代历史

### 迭代 1（失败）

最初使用 closure 方式传递 `negative_slope`：

```python
# kernels/leaky_relu.py（第 1 版 — 错误）
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ntops.kernels.element_wise import arrangement

def _make_application(negative_slope):
    def application(input, output):
        output = ntl.where(input >= 0, input, negative_slope * input)  # noqa: F841
    return application
```

**错误**：`NameError: 'negative_slope' is not defined`

**根因**：NineToothed 通过源码检查编译 application 函数。闭包捕获的 `negative_slope` 在 triton 编译时不可见。

### 迭代 2-3（失败）

尝试用 `Tensor(0, dtype=ninetoothed.float64)` 传标量：

```python
tensors = (
    Tensor(ndim, dtype=dtype),
    Tensor(ndim, dtype=dtype),
    Tensor(0, dtype=ninetoothed.float64),  # ⚠️ 问题所在
)
```

**错误**：`IncompatibleTypeError: pointer<fp64> and float32`

**根因**：`Tensor(0, dtype=ninetoothed.float64)` 生成 fp64 指针，与 fp32 输入做乘法时类型不兼容。

### 迭代 4（成功）— 最终版本

使用 `Tensor(0, constexpr=True, value=negative_slope)`：

```python
# kernels/leaky_relu.py（最终版）
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
    kernel(input, output, negative_slope)
    return output
```

## 阶段 4：精度验证

```
test_leaky_relu_float32_basic PASSED
test_leaky_relu_float16_basic PASSED
test_leaky_relu_float32_large PASSED
test_leaky_relu_float16_large PASSED
test_leaky_relu_custom_slope PASSED
test_leaky_relu_edge_cases PASSED
```

所有四项必检通过（allclose + 无 NaN + 无 Inf）。

## 阶段 5：性能评估

| 算子 | 输入规模 | ntops | PyTorch | 比率 |
|------|----------|-------|---------|------|
| leaky_relu | 4096x4096 | 0.520ms | 0.478ms | 0.92x |

六项策略评估：
1. ✅ 内存访问模式：element-wise tile 保证 coalesced access
2. ⬜ 算子融合：单算子无融合空间
3. ⬜ 循环展开：无循环
4. ⬜ 减少同步：无同步点
5. ✅ 精度策略：constexpr 不影响精度
6. ⬜ 计算重组：`ntl.where` 已是最优表达

## 关键经验

**标量参数传递**：`Tensor(0, constexpr=True, value=...)` 是传递标量系数的推荐方式。
- 不要使用闭包（编译时不可见）
- 不要使用 `Tensor(0, dtype=ninetoothed.float64)`（类型不兼容）
- constexpr Tensor 仍需在 kernel 调用时传入参数
