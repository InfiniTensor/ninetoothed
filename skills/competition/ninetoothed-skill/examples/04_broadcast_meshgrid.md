# 示例 4：广播算子开发 — meshgrid

> 本示例展示需要 1D→2D 广播的算子如何完成开发。meshgrid 的输入是
> 两个 1D tensor，输出是两个 2D tensor，通过自定义 arrangement 实现广播。

## 任务输入

```
CPU 参考实现：
import torch

def meshgrid(x, y):
    nx = len(x)
    ny = len(y)
    X = x.unsqueeze(0).expand(ny, nx).clone()
    Y = y.unsqueeze(1).expand(ny, nx).clone()
    return X, Y
```

## 阶段 1：分析

- **算子类型**：Broadcast / 1D→2D（无现成共享 arrangement，需要自定义 arrangement）
- **输入**：`x` 形状 `(M,)`，`y` 形状 `(N,)` — 两个 1D tensor，长度可以不同
- **输出**：`X` 形状 `(N, M)`，`Y` 形状 `(N, M)` — 两个 2D tensor
- **核心挑战**：1D 输入广播到 2D 输出，需要自定义 tile + expand arrangement
- **特殊参数**：两个独立的 block_size（M 方向和 N 方向）
- **共享 arrangement**：无 — 需要自定义

## 阶段 2：生成

### Arrangement 设计

meshgrid 的核心是 1D→2D 广播：
- `x[1,i]` → 在 M 方向保持不变（tile 到 `(1, block_n)`），在 N 方向广播（expand 到 `(block_m, -1)`）
- `y[j,1]` → 在 N 方向保持不变（tile 到 `(block_m, 1)`），在 M 方向广播（expand 到 `(-1, block_n)`）

```
arrangement 操作链：
x(1,N) → tile((1, block_n)) → expand((block_m, -1))
y(M,1) → tile((block_m, 1)) → expand((-1, block_n))
X(M,N) → tile((block_m, block_n))
Y(M,N) → tile((block_m, block_n))
```

### Kernel 文件

```python
# kernels/meshgrid.py
import functools

import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()


def arrangement(x, y, X, Y, block_size_m=None, block_size_n=None):
    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N

    X_arranged = X.tile((block_size_m, block_size_n))
    Y_arranged = Y.tile((block_size_m, block_size_n))

    x_arranged = x.tile((1, block_size_n))
    x_arranged = x_arranged.expand((X_arranged.shape[0], -1))

    y_arranged = y.tile((block_size_m, 1))
    y_arranged = y_arranged.expand((-1, Y_arranged.shape[1]))

    return x_arranged, y_arranged, X_arranged, Y_arranged


def application(x, y, X, Y):
    X = x  # noqa: F841
    Y = y  # noqa: F841


def premake(ndim, dtype=None, block_size_m=None, block_size_n=None):
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
```

### Torch 文件

```python
# torch/meshgrid.py
import torch

import ntops
from ntops.torch.utils import _cached_make


def meshgrid(x, y):
    nx = x.shape[0]
    ny = y.shape[0]

    x_2d = x.unsqueeze(0)
    y_2d = y.unsqueeze(1)

    X = torch.empty(ny, nx, dtype=x.dtype, device=x.device)
    Y = torch.empty(ny, nx, dtype=y.dtype, device=y.device)

    kernel = _cached_make(ntops.kernels.meshgrid.premake, 2)

    kernel(x_2d, y_2d, X, Y)

    return X, Y
```

### 关键设计决策

1. **Torch 层做 unsqueeze**：`x.unsqueeze(0)` 将 `(M,)` 变为 `(1, M)`，`y.unsqueeze(1)` 将 `(N,)` 变为 `(N, 1)`。这样输入就变成了 2D tensor，arrangement 中的 ndim 固定为 2
2. **双 block_size**：M 和 N 方向可以独立调优，分别用 `ninetoothed.block_size()` 自动搜索
3. **tile + expand 广播**：x 在 N 方向 tile（block_size_n），在 M 方向 expand（block_size_m）。-1 表示保持原大小。这避免了在 GPU 上做显式的内存复制
4. **application 即恒等映射**：`X = x; Y = y` — 实际的计算已经在 arrangement 的 tile/expand 中完成了广播，application 只需要做 identity copy

## 阶段 4：精度验证

```
test_meshgrid_basic[float32] PASSED
test_meshgrid_basic[float16] PASSED
test_meshgrid_large[float32] PASSED
test_meshgrid_large[float16] PASSED
test_meshgrid_edge_cases PASSED
```

四项必检全部通过，包括非整除 block_size 的边界情况（17×13）。

## 阶段 5：性能评估

六项策略评估：
1. ✅ 内存访问模式：tile 后 expand 是零成本广播（无额外内存读写），coalesced access
2. ⬜ 算子融合：meshgrid 本身是纯内存操作，无计算可融合
3. ⬜ 循环展开：application 只有恒等赋值，无循环
4. ⬜ 减少同步：单 kernel launch
5. ✅ 精度策略：identity 操作无精度损失
6. ⬜ 计算重组：不适用（纯内存操作）

## 关键经验

**自定义 Arrangement 的设计思路**：
1. **先确定输出 tile**：输出分成 `(block_m, block_n)` 的块
2. **从输出反推输入**：每个输入 tile 对应输入的哪部分？用 tile + expand 描述
3. **tile(N, -1) + expand** 是广播的核心模式：tile 指定该维度在输入中的分块方式，expand(-1, ...) 广播到输出大小
4. **Torch 层负责维度预处理**：unsqueeze/flatten 等元操作在 host 端做，arrangement 只关心 GPU tile 映射

**与 matmul arrangement 的对比**：
- matmul 使用 tile → expand → squeeze 做 K 维度的规约
- meshgrid 使用 tile → expand 做 N/M 维度的广播（不需要 squeeze）
- 同一个 expand 元操作，方向不同就实现了不同的语义（规约 vs 广播）
