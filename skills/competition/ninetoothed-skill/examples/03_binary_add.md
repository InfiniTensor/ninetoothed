# 示例 3：Binary 算子开发 — add

> 本示例展示 Element-wise Binary 算子如何完成开发。add 是最简单的二元算子，
> 额外包含 `alpha` 标量系数参数，展示运行时标量的正确传递方式。

## 任务输入

```
CPU 参考实现：
import torch

def add(input, other, alpha=1):
    return input + alpha * other
```

## 阶段 1：分析

- **算子类型**：Element-wise Binary（模式 1，共享 element_wise arrangement）
- **输入**：2 个 N 维 tensor（同 shape），1 个标量系数 `alpha`
- **输出**：1 个 N 维 tensor（与输入同 shape/dtype）
- **特殊参数**：`alpha`（运行时标量，默认 1，参与逐元素运算）
- **共享 arrangement**：`kernels/element_wise.py`

## 阶段 2：生成

### 标量参数选择决策

`alpha` 与 `input`/`other` 做逐元素乘法后加法 → 属于"运行时标量与输入做逐元素运算" →
选用 **运行时同类型 Tensor**（`Tensor(0, dtype=...)`），而非 constexpr。

> 对比 leaky_relu 的 `negative_slope`：它是编译时确定的条件系数，适合 constexpr。
> 而 add 的 `alpha` 是 runtime 可变值，适合 0-dim Tensor。

### Kernel 文件

```python
# kernels/add.py
import functools

import ninetoothed
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, other, alpha, output):
    output = input + alpha * other  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
```

### Torch 文件

```python
# torch/add.py
import torch

import ntops
from ntops.torch.utils import _cached_make


def add(input, other, *, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.add.premake, input.ndim)

    kernel(input, other, alpha, out)

    return out
```

### 关键规则

1. **alpha 是 0-dim Tensor**：`Tensor(0, dtype=ninetoothed.float64)` —
   与 leaky_relu 不同，alpha 不是 constexpr，因为它是运行时可变的值
2. **输出分配**：`torch.empty_like(input)` 简化 shape/dtype/device 设置
3. **参数传参**：`kernel(input, other, alpha, out)` — 按 premake 中 Tensor 声明顺序传入

## 阶段 4：精度验证

```
test_add[float32] PASSED
test_add[float16] PASSED
```

四项必检全部通过（allclose + 无 NaN + 无 Inf）。

## 阶段 5：性能评估

| 算子 | 输入规模 | ntops | PyTorch | 比率 |
|------|----------|-------|---------|------|
| add | 4096x4096 | 0.22ms | 0.21ms | 0.87x |

六项策略评估：
1. ✅ 内存访问模式：element-wise tile 保证 coalesced access
2. ⬜ 算子融合：`alpha * other + input` 本身已经是线程内融合
3. ⬜ 循环展开：无循环
4. ⬜ 减少同步：无同步点
5. ✅ 精度策略：alpha 的 float64 与 float32 的乘法由 ntl 自动处理精度提升
6. ⬜ 计算重组：已是最简形式 `input + alpha * other`

## 关键经验

**Binary 算子的模式**：
- 两个同 shape 的输入 tensor + 一个输出 tensor + 可选的运行时标量
- 与 Unary element-wise 共享同一个 `element_wise` arrangement
- arrangement 自动处理任意 ndim 的 flatten + tile
- 运行时标量用 `Tensor(0, dtype=...)`，编译时常量用 `Tensor(0, constexpr=True, value=...)`
