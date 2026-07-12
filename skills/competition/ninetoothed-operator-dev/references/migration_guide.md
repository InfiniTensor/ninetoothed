# Triton 到九齿迁移指南

## 1. 迁移概述

将 Triton kernel 迁移到九齿的关键，不是逐行 API 替换，而是**范式转换**：

| 概念 | Triton 方式 | 九齿方式 |
|------|-----------|---------|
| 并行分解 | 手动 `pid → block_start → offsets → mask` | `tile()` 一行表达 |
| 内存管理 | 显式 `tl.load` / `tl.store` | `make()` 自动生成 |
| 块大小 | `tl.constexpr` | `Symbol(constexpr=True)` 或 `block_size()` |
| 内核定义 | `@triton.jit` 装饰器 | `ninetoothed.make(arrangement, application, tensors)` |

迁移步骤：

1. 分析 Triton kernel 的访存模式和并行策略
2. 将 `pid`/`offsets`/`mask` 逻辑映射为 `arrangement` 中的 `tile()` 调用
3. 将计算逻辑写入 `application` 函数
4. 用 `ninetoothed.make()` 组装内核
5. 对比 PyTorch 验证正确性和性能

## 2. 原语对照表

> ⚠️ `ntl.*` 函数是代码生成器在编译时解析的符号名，Python 运行时可能不存在对应的函数对象。以下仅列出在 `operators/` 中实际使用并通过 correctness 测试的 API。

### 2.1 数学运算（已验证 ✅）

| Triton | 九齿 | 使用位置 |
|--------|------|---------|
| `tl.exp` | `ntl.exp` | gelu.py, softmax.py（需 fp32 输入） |
| `tl.sigmoid` | `ntl.sigmoid` | sigmoid.py, gelu.py（需 fp32 输入） |
| `tl.maximum` | `ntl.maximum` | relu.py |

> ⚠️ `ntl.sqrt`、`ntl.abs`、`ntl.minimum` 在 SKILL.md 中提及但未在算子中使用——可能工作但未确认。

### 2.2 归约运算（已验证 ✅）

| Triton | 九齿 | 使用位置 |
|--------|------|---------|
| `tl.sum` | `ntl.sum` | softmax.py, sum.py, rms_norm.py |
| `tl.max` | `ntl.max` | softmax.py |

> ⚠️ `ntl.sum` / `ntl.max` 对当前 tile 内所有元素归约（axis=None）。无法指定归约维度。

### 2.3 类型转换（已验证 ✅）

| Triton | 九齿 | 使用位置 |
|--------|------|---------|
| `x.to(tl.float32)` | `ntl.cast(x, ntl.float32)` | gelu.py, sigmoid.py, rms_norm.py |

### 2.4 并行与分块

这是迁移中最核心的概念转换：

| Triton | 九齿 | 说明 |
|--------|------|------|
| `pid = tl.program_id(axis=0)` | 隐含在 `tile()` 中 | 自动分配各 block |
| `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` | 隐含在 `tile()` 中 | 自动生成 thread offset |
| `mask = offsets < n_elements` | 隐含在 `tile()` 中 | 自动处理边界 |
| `tl.load(x_ptr + offsets, mask=mask)` | 隐含在 `make()` 中 | arrangement 定义后自动生成 load |
| `tl.store(output_ptr + offsets, output, mask=mask)` | 隐含在 `make()` 中 | arrangement 定义后自动生成 store |
| `BLOCK_SIZE: tl.constexpr` | `Symbol("BLOCK_SIZE", constexpr=True)` | 编译时常量。逐元素也可用 `block_size()` 自动调优 |

## 3. 完整迁移示例：向量加法

### 3.1 Triton 原始代码

```python
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

### 3.2 九齿迁移版本

```python
import ninetoothed
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(x, y, output, BLOCK_SIZE=BLOCK_SIZE):
    """Triton 的 pid→offsets→load 在此压缩为一行 tile()"""
    return (
        x.tile((BLOCK_SIZE,)),
        y.tile((BLOCK_SIZE,)),
        output.tile((BLOCK_SIZE,)),
    )

def application(x, y, output):
    """逐元素计算与 Triton 完全一致"""
    output = x + y  # noqa: F841

def create_add_kernel():
    return ninetoothed.make(arrangement, application,
                          (Tensor(1), Tensor(1), Tensor(1)))
```

### 3.3 对比总结

| 指标 | Triton | 九齿 |
|------|--------|------|
| 代码行数 | 11 行 | 12 行（含 create 工厂函数） |
| Load/Store 管理 | 手动 | 自动（make 生成） |
| 边界处理 | 手动 mask | 自动 |
| 并行分解 | 手动 pid + offset | tile 抽象 |
| BLOCK_SIZE 配置 | `tl.constexpr` | `Symbol(constexpr=True)` 或 `block_size()` |

## 4. 迁移检查清单

- [ ] 识别所有 `tl.program_id` / `tl.arange` 使用点 → 映射为 tile 形状
- [ ] 将计算逻辑从 Triton kernel body 迁移到 `application` 函数
- [ ] 确定 BLOCK_SIZE 类型：归约算子 → `Symbol(constexpr=True)`；逐元素 → `block_size()` 或 `Symbol(constexpr=True)`
- [ ] fp16 输入的计算确保用 `ntl.cast(x, ntl.float32)` 转换精度
- [ ] `application` 中的输出赋值加 `# noqa: F841`
- [ ] `Tensor(n)` 维度数 = `len(tile_shape)`
- [ ] 对比 PyTorch 验证正确性（`torch.allclose`）
- [ ] benchmark 对比性能

## 5. 注意事项

### 5.1 不直接暴露的 Triton 原语

九齿**不暴露**以下概念，由 `make()` 的代码生成器自动处理：

- `tl.program_id` / `tl.num_programs` → tile 自动分配
- `tl.arange` → tile 内自动生成
- `tl.load` / `tl.store` → arrangement 定义后自动生成
- 边界 mask → tile 自动处理

如果你的 Triton kernel 重度依赖这些原语做复杂控制流（如非规则访存、动态 mask），九齿可能不适合直接迁移，需要重新设计 arrangement。

### 5.2 复杂归约

`ntl.sum` / `ntl.max` 对 tile 内所有元素归约（无 axis 参数）。对于需要指定归约维度的场景（如 softmax 沿 dim=-1），安排 tile shape 使得归约维度 = BLOCK_SIZE、保留维度 = 1。

### 5.3 libdevice 限制

`ntl.libdevice.*` 路径在代码生成中不可用。如需 libdevice 函数（tanh, erf 等），必须：

```python
from ninetoothed.language import libdevice  # ✅ 直接导入为独立全局变量
libdevice.tanh(x)                            # ✅

# ❌ ntl.libdevice.tanh(x)  — 代码生成失败！
```

或用恒等式替代（如 `tanh(x) = 2*sigmoid(2x) - 1`）。

### 5.4 迁移时机

- **适合迁移**：逐元素、简单归约、融合算子
- **不适合迁移**：需要手动 scratch buffer 的算子（矩阵乘法、卷积）、动态 shape、需要跨 block 同步的复杂归约

## 6. 参考

- SKILL.md §2-3：完整的 tile 决策表和编码模式
- `operators/` 目录：9 个已验证的九齿算子实现
- `tests/self_eval_5_triton_migrate_add.md`：迁移过程的详细诊断记录
