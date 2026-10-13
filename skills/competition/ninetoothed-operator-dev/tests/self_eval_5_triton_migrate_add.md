# 自测 5：Triton Add 迁移（迁移/诊断类）

## 实验环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| CUDA | 12.8 |
| PyTorch | 2.8.0+cu128 |
| Triton | 3.4.0 |
| ninetoothed | 0.26.0 |
| Python | 3.12 |
| AI 智能体 | DeepSeek-V4-Pro |
| dtype | float16 |
| 测试日期 | 2026-07-12 |

## 任务说明

将一段标准 Triton 向量加法 kernel 迁移为九齿实现，通过原语对照表完成手动改写，验证 correctness 和性能。

## 原始 Triton 代码

```python
# triton_add.py
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

## 迁移过程

### 步骤 1: 识别 Triton 原语并映射

Triton 中显式管理的概念在九齿中由 `make()` 自动处理：

| Triton 原语 | 九齿等价 | 说明 |
|------------|---------|------|
| `pid = tl.program_id(0)` | 隐含在 `tile` 中 | `tile((BLOCK_SIZE,))` 自动分配各 block |
| `tl.arange(0, BLOCK_SIZE)` | 隐含 | tile 内自动生成 thread offset |
| `tl.load(...)` | 隐含在 `make()` 中 | arrangement 定义后自动生成 load |
| `tl.store(...)` | 隐含在 `make()` 中 | arrangement 定义后自动生成 store |
| `mask = offsets < n` | 隐含 | tile 自动处理边界（无效位置跳过） |
| `output = x + y` | `output = x + y` | 逐元素计算一致 |
| `BLOCK_SIZE: tl.constexpr` | `Symbol("BLOCK_SIZE", constexpr=True)` | 编译时常量 |

### 步骤 2: 编写九齿 Arrangement

Triton 中手动管理 `pid → block_start → offsets`，九齿中用 `tile` 一行表达:

```python
def arrangement(x, y, output, BLOCK_SIZE=BLOCK_SIZE):
    return (
        x.tile((BLOCK_SIZE,)),
        y.tile((BLOCK_SIZE,)),
        output.tile((BLOCK_SIZE,)),
    )
```

每个 tensor 沿 BLOCK_SIZE 维度分块，ninetoothed 自动处理并行分配、边界 mask 和 data load/store。

### 步骤 3: 编写 Application

```python
def application(x, y, output):
    output = x + y  # noqa: F841
```

与 Triton 的 `output = x + y` 完全一致。

### 步骤 4: 组装 Kernel

```python
import ninetoothed
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def create_add_kernel():
    return ninetoothed.make(arrangement, application,
                          (Tensor(1), Tensor(1), Tensor(1)))
```

## 迁移总结

| 指标 | Triton | 九齿 | 变化 |
|------|--------|------|------|
| 代码行数 | 11 行 | 12 行 | 持平 |
| Load/store 管理 | 手动 | 自动（make 生成） | 减少 |
| 边界处理 | 手动 mask | 自动 | 减少 |
| 并行分解 | 手动 pid + offset | tile 抽象 | 简化 |
| BLOCK_SIZE 配置 | `tl.constexpr` | `Symbol(constexpr=True)` | 等价 |

**关键差异**: Triton 显式管理内存和并行，九齿将其抽象为元操作层（tile），减少手动编码但增加概念学习成本。

## Correctness 测试

迁移后的九齿 add kernel 通过全部 3 个规模测试（100, 512, 1024），与 PyTorch `x + y` 结果一致（`atol=1e-5, rtol=1e-3`）。

```
Shape (100,)         : ✓ 通过
Shape (512,)         : ✓ 通过
Shape (1024,)        : ✓ 通过
```

## 性能验证

在 RTX 4090 上，迁移后的九齿 Add kernel vs PyTorch：

| size | ninetoothed (ms) | torch (ms) | 比值 |
|------|---------------------|------------|------|
| 1024 | 0.003 | 0.005 | 0.60x |
| 8192 | 0.004 | 0.006 | 0.67x |
| 65536 | 0.004 | 0.006 | 0.67x |

迁移后的九齿版本在所有规模上均快于 PyTorch，说明 tile 抽象未引入额外性能开销。

## Generated Source 验证

可通过 `kernel.src` 查看生成的 Triton 代码，验证其与原始 Triton 实现的等价性：

```python
from operators import create_add_kernel
k = create_add_kernel()
print(k.src)  # 打印生成的 Triton 代码
```

也可查看缓存文件：

```bash
ls -lt ~/.ninetoothed/*.py | head -5
```

## 注意事项

- 九齿不暴露 `pid` / `tl.arange` / `mask` ——这些由 `make()` 的代码生成器自动处理
- 复杂归约（如 `tl.atomic_add`）在九齿中用 `ntl.sum` 等高层原语替代
- 迁移 workflow: 分析 Triton 结构 → 识别隐式抽象 → 编写九齿代码 → correctness 验证 → benchmark 对比 → generated source 审查
- 不适合迁移的场景：需要手动 scratch buffer（矩阵乘法）、动态 shape、跨 block 同步
