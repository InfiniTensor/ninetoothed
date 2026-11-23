---
description: >
  NineToothed GPU 算子开发完整指导。从 CPU 参考实现出发，指导 AI 智能体完成
  分析、代码生成、编译调优、精度验证、性能优化、错误诊断与修复的完整闭环。
triggers:
  - ninetoothed
  - operator
  - 算子
  - kernel
  - ntops
  - arrangement
  - application
---

# NineToothed 算子开发 Skill

你是 NineToothed GPU 算子开发的完整指导者。你管理从 CPU 参考实现到 GPU 算子的完整生命周期。

**核心原则：CPU 实现是唯一的精度基准。代码不会说谎。**

## 项目路径

| 路径 | 说明 |
|------|------|
| `$PROJECT_ROOT/` | 项目根目录（包含 ntops 和 NineToothed） |
| `$PROJECT_ROOT/ntops/` | ntops 算子库 |
| `$PROJECT_ROOT/NineToothed/` | NineToothed DSL 框架 |

> **确定 `$PROJECT_ROOT` 的方法**（按顺序尝试）：
> 1. 检查当前工作目录下是否存在 `ntops/src/ntops/kernels/` 和 `NineToothed/src/ninetoothed/` 子目录
> 2. 搜索 `ntops.kernels` 可导入的 Python 包位置：`python -c "import ntops.kernels; print(ntops.kernels.__path__[0])"`，取 `/ntops/` 的父目录
> 3. 搜索 `ninetoothed` 包的安装位置：`python -c "import ninetoothed; print(ninetoothed.__path__[0])"`，取 `/NineToothed/` 的父目录
> 4. 若均失败，询问用户提供路径

---

## 一、触发场景

当用户提出以下类型任务时激活本 Skill：
1. 给出一段 CPU 参考实现（NumPy/PyTorch），要求生成 GPU 算子
2. 要求为 NineToothed 生态编写新的 arrangement/application
3. 要求调试、优化、测试已有的九齿算子
4. 要求进行算子性能对比或回退分析
5. 要求处理非连续输入、步长、偏移等边界场景

### 输入格式

用户提供一段 **CPU 参考实现**，可以是：
- Python 函数（使用 NumPy 或 PyTorch）
- 算子描述 + 算法伪代码

示例：
```python
import numpy as np

def silu(x):
    return x / (1 + np.exp(-x))
```

### 输出格式

一份完整报告，包含：
1. 算子名称和分类
2. 生成的文件路径（kernels/ + torch/）
3. 精度验证结果（每种 dtype 的 allclose / NaN / Inf 检查）
4. 性能评估（如有对比数据）
5. 处理的边界情况

---

## 二、能力边界

### NineToothed 能做的事

| 模式 | 名称 | 复杂度 | 共享 Arrangement | 典型算子 |
|------|------|--------|------------------|----------|
| 1 | Element-wise | 低 | `kernels/element_wise.py` | relu, silu, add, mul, sigmoid, neg, abs, leaky_relu |
| 2 | Reduction | 中 | `kernels/reduction.py` | softmax, rms_norm, layer_norm, log_softmax |
| 3 | Matmul | 高 | `kernels/mm.py` | mm |
| 4 | Batched Matmul | 高 | `kernels/bmm.py` | bmm |
| 5 | Addmm（组合 mm） | 高 | 复用 mm | addmm |
| 6 | Conv2D（im2col + mm） | 很高 | 复用 mm | conv2d |
| 7 | Attention | 很高 | `kernels/scaled_dot_product_attention.py` | scaled_dot_product_attention |
| 8 | Pooling | 中 | `kernels/pooling.py` | max_pool2d, avg_pool2d |
| 9 | RoPE | 中 | `kernels/rotary_position_embedding.py` | rotary_position_embedding |

**可用元操作**：tile, expand, squeeze, unsqueeze, permute, flatten, ravel, pad, indexing

**ntl 操作**：zeros, full, dot, sum, max, exp, exp2, where, cast/to, trans, maximum, minimum, rand, load, store, atomic_add, arange, program_id, libdevice.*

**数据类型**：float16, bfloat16, float32, float64, int8/16/32/64, uint8/16/32/64

### 九种模式分类决策树

```
CPU 代码是否包含点积累加（矩阵乘法）？
├── 是 → 输入是 3D 吗？
│   ├── 是 → 模式 4（Batched Matmul）
│   └── 否 → 是否有 bias/alpha/beta 参数？
│       ├── 是 → 模式 5（Addmm）
│       └── 否 → 模式 3（Matmul）
└── 否 → CPU 代码是否包含空间维度（H, W）+ kernel/stride/padding？
    ├── 是 → 是否有权重 tensor 做卷积？
    │   ├── 是 → 模式 6（Conv2D）
    │   └── 否 → 模式 8（Pooling）
    └── 否 → CPU 代码是否沿某个维度做规约（sum/max/normalize）？
        ├── 是 → 模式 2（Reduction）
        └── 否 → 是否有 query/key/value 结构？
            ├── 是 → 模式 7（Attention）
            └── 否 → 是否用 sin/cos 做旋转？
                ├── 是 → 模式 9（RoPE）
                └── 否 → 模式 1（Element-wise）
```

### application 函数体内的绝对禁止

1. `torch.*` — torch 是 CPU/host 端库
2. `triton.*` — 通过 ntl 间接使用
3. `cuda.*` 或原始 CUDA 代码
4. 原始指针运算（用 `ntl.load`/`ntl.store`）
5. 动态内存分配
6. 递归
7. 数据依赖的 while 循环（range 循环可以）
8. import 任何模块
9. print/IO 操作
10. Python list/dict 等动态容器
11. **闭包/外部变量引用** — NineToothed 通过源码检查编译 application，闭包变量在 triton 编译时不可见

---

## 三、六阶段工作流

```
① 分析 CPU 实现 → ② 生成初始算子 → ③ 自动调优 → ④ 精度验证 → ⑤ 性能优化 → ⑥ 输出报告
```

### 阶段 1：分析 CPU 实现

1. 读取用户提供的 CPU 参考实现
2. 使用决策树判断算子类型（9 种模式之一）
3. 提取关键信息：
   - 输入/输出 tensor 的维度（ndim）
   - 数据类型要求
   - 特殊参数（标量、constexpr、padding、strides、dilation）
   - 边界情况（None 值、空 tensor、广播）
4. **检查 `ntl.libdevice` 是否有现成实现**：`python -c "from ninetoothed.language import libdevice; print('funcname' in dir(libdevice))"`
   - CUDA libdevice 提供 150+ 数学函数（`pow`, `sin`, `tanh`, `nextafter`, `copysign`...）
   - 如果 libdevice 有 → 直接用，**不要重新实现**（一条 `libdevice.xxx()` 替代整个位操作算法）
   - 注意 dtype 精度：libdevice 函数默认为 double，float32 需验证 subnormal 等边界
5. 确定是复用已有 arrangement 还是需要新的
6. **是否需要 GPU kernel？** — 判断优先级：
   ① NineToothed kernel 可实现（优先）：eye（program_id 对角）、chunk/unbind（identity per slice）、repeat（broadcast）
   ② 可变数量输出：premake 的 Tensor 数量固定，需循环调用 kernel（每 chunk/slice 一次 launch）
   ③ 张量创建（无输入）：使用 `ntl.program_id` + `ntl.arange` + `ntl.where` 可在 kernel 内创建（如 eye）
7. **依赖分析**：CPU 实现是否调用了其他函数（如 lcm 调用了 gcd）？
   - 被调函数在 ntops 中**已有实现** → torch 层复用 `ntops.torch.xxx`，两个 kernel launch
   - 被调函数**没有实现** → kernel 内联，避免额外的 launch + 内存往返
   - 判断依据：检查 `ntops/torch/__init__.py` 的 `__all__` 列表，搜索 `ntops.kernels` 目录

### 阶段 2：生成初始算子

根据分类结果，选择对应的代码模板生成代码。详细模板见 `references/code_templates.md`。

生成前需明确以下信息：
1. CPU 参考代码
2. 分类结果（模式编号）
3. 要使用的模板
4. Tensor 参数（ndim, dtype, 标量参数, padding 值等）
5. 特殊处理说明

**生成的文件**：
- `ntops/src/ntops/kernels/{op_name}.py` — arrangement + application + premake
- `ntops/src/ntops/torch/{op_name}.py` — PyTorch 接口层
- 更新 `ntops/src/ntops/kernels/__init__.py`（添加 import + `__all__` 条目）
- 更新 `ntops/src/ntops/torch/__init__.py`（添加 import + `__all__` 条目）

**核心规则**：

1. **Kernel 文件允许的 import**：`functools`, `enum`, `math`, `copy`, `ninetoothed`, `ninetoothed.language`, `ninetoothed.Tensor/Symbol/block_size`, `ntops.kernels.*`
2. **Torch 文件允许的 import**：`torch`, `ntops`, `_cached_make`, `_get_matmul_input_precision`
3. **Application 输出赋值**必须有 `# noqa: F841`
4. **浮点累加**始终用 float32
5. **premake** 返回 `(arrangement_, application, tensors)` 三元组
6. **premake** 使用 `functools.partial` 绑定 arrangement 参数

**标量参数传递规则（关键）**：

| 场景 | 方式 | 声明 | 传递 |
|------|------|------|------|
| 标量与输入逐元素运算（clamp） | 运行时同类型 Tensor | `Tensor(ndim, dtype=dtype)` | torch 层直接传值 |
| 条件系数/枚举/编译时确定（leaky_relu slope） | **constexpr Tensor** | `Tensor(0, constexpr=True, value=默认值)` | kernel 调用时仍需传入 |
| 整数枚举常量 | constexpr | `Tensor(0, constexpr=True, value=enum_value)` | 同上 |
| 运行时浮点系数（add alpha） | 0-dim Tensor | `Tensor(0, dtype=ninetoothed.float64)` | torch 层直接传值 |

> **标量参数选择原则**（按优先级）：
> 1. **编译时常量 → constexpr**（最安全，类型自动适配）
> 2. **运行时标量需与输入逐元素运算 → `Tensor(ndim, dtype=dtype)`**（clamp 模式，类型安全）
> 3. **运行时浮点系数仅做乘法 → `Tensor(0, dtype=ninetoothed.float64)`**（add alpha 模式，注意：此方式在条件分支如 `ntl.where` 中可能触发 `IncompatibleTypeError`）
>
> **禁止**：在 application 内使用闭包/外部变量。NineToothed 通过源码检查编译，闭包变量不可见。
> **所有参数必须通过 application 的函数参数或 Tensor 声明传入。**

**Application 编写关键模式**：

```python
# 1. 浮点累加 — 始终用 float32
accumulator = ntl.zeros(shape, dtype=ntl.float32)

# 2. 输出前转回原始 dtype
output = ntl.cast(result, output.dtype.dtype)  # noqa: F841

# 3. 边界保护 — padded tensor 用 ntl.where
value = ntl.where(tensor.offsets(-1) < tensor.source.shape[-1], value, 0)

# 4. float16 安全 exp/log — 先转 float32
ntl.exp(ntl.cast(x, ntl.float32))
ntl.log(ntl.cast(x, ntl.float32))

# 5. 全局位置感知 — ntl.program_id + ntl.arange
pid = ntl.program_id(0)                    # 当前 tile 的 block 索引
j = ntl.arange(0, output.shape[0])          # tile 内偏移 [0, 1, ..., block_size-1]
global_idx = pid * output.shape[0] + j      # 全局扁平化索引
row = global_idx // n_cols                   # 矩阵行号
col = global_idx % n_cols                    # 矩阵列号
output = ntl.where(row == col, 1, 0)        # 对角线检测（eye 算子）

# 6. 张量创建 — ntl.full 可在 application 内创建常量张量
ntl.full(shape, value, dtype=ntl.float32)

# 7. Triton 原语访问 — 通过全限定名 ninetoothed.language.xxx
# ntl 模块只暴露 libdevice，但 code generator 会将
# ninetoothed.language.xxx 自动转换为 triton.language.xxx
import ninetoothed.language  # 必须导入
val = ninetoothed.language.cast(input, ninetoothed.language.int32)
counts = ninetoothed.language.histogram(val, num_bins=256)  # 整数直方图
# ninetoothed.language.atomic_add(ptr, val)  # 原子加
# ninetoothed.language.gather(src, indices)  # 按索引收集
```

**Torch 层编写规则**：

1. **必须用 `_cached_make`** 编译 kernel（自动缓存）
2. **处理 None 默认值**：检查 `out=None`、`bias=None` 等参数
3. **输出分配**：
   ```python
   output = torch.empty(shape, dtype=input.dtype, device=input.device)
   ```
4. **mm 系列**：使用 `_get_matmul_input_precision()` 获取精度设置
5. **参数传递**：torch 层的参数按 premake 定义的顺序传递给 kernel

**Kernel 文件允许的 import**：
```python
# 允许
import functools, enum, math, copy
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, Symbol, block_size
from ntops.kernels.element_wise import arrangement    # element-wise
from ntops.kernels.reduction import arrangement       # reduction
from ntops.kernels.pooling import arrangement         # pooling
from ntops.kernels import mm, bmm                    # 复用 matmul

# 禁止
# import torch   ← 禁止！
# import triton  ← 禁止！
# import cuda    ← 禁止！
# import numpy   ← 禁止！
```

### 阶段 3：自动调优

1. 验证生成的 kernel 能编译通过
2. 如果编译失败，诊断错误并修复后才继续
3. `ninetoothed.make()` 内置 auto-tune，block_size 会自动搜索最优值

### 阶段 4：精度验证

生成测试脚本并运行。精度标准：

| 数据类型 | rtol | atol |
|----------|------|------|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1e-2 | 1e-2 |
| int32/int64 | 0 | 0（精确匹配） |

**四项必检**：
1. `torch.allclose(output, reference, rtol=rtol, atol=atol)` 通过
2. 无 NaN：`assert not torch.isnan(output).any()`
3. 无 Inf：`assert not torch.isinf(output).any()`
4. 整数类型精确匹配：`assert torch.equal(output, reference)`

测试必须覆盖：
- float32 和 float16 两种数据类型
- 不同输入规模（小/中/大）
- 边界情况（size 不整除 block_size）

### 阶段 5：性能 Benchmark

精度验证通过后，对算子进行 benchmark。**不可跳过。**

**Benchmark 代码模板**：

```python
import torch

def bench(fn, warmup=10, repeat=100):
    """CUDA events 计时，warmup + 多次重复取平均"""
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat): fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat
```

**Benchmark 规范**：

| 项目 | 要求 |
|------|------|
| 基线 | PyTorch 同功能 API |
| 规模 | 至少 3 种：小（256²）、中（1024²）、大（4096²） |
| dtype | float32（通用）、float16（如适用）、int64（整数算子） |
| 预热 | 10 次 |
| 测量 | 100 次取平均 |
| 同步 | 每次测量前后 `torch.cuda.synchronize()` |

**性能判定标准**：

| 比率 (ntops/PyTorch) | 判定 | 处理 |
|:--:|------|------|
| ≤ 1.2x | OK | 通过 |
| 1.2x – 4x | SLOW | 启动六项策略诊断，记录原因 |
| > 4x | GAP | 必须分析根因，如属先天限制（view vs copy）则明确记录 |

### 阶段 5b：性能优化与迭代

**六项策略**（按顺序逐项评估，每项必须标注结论）：

| 优先级 | 策略 | 评估方法 |
|--------|------|----------|
| 1 | 内存访问模式优化 | 检查 tile shape 是否保证 coalesced access |
| 2 | 算子融合 | 检查是否可以合并相邻操作减少 kernel launch。**设计时决策**：若算子内部依赖其他计算（如 lcm 需要 gcd），在 Stage 1 就应判断依赖是否已有实现——有则复用，无则内联，避免事后再重构 |
| 3 | 循环展开 | 检查 application 中小循环是否可以手动展开 |
| 4 | 减少同步开销 | 检查是否有不必要的同步点、多次 kernel launch |
| 5 | 精度策略调整 | 检查中间计算是否可用 float32 累加 |
| 6 | 计算重组 | 检查是否可用 exp2 trick、log-sum-exp 等技巧 |

**特殊性能模式**（需明确识别和记录）：

| 模式 | 症状 | 原因 | 处理 |
|------|------|------|------|
| launch overhead | 小规模 >4x，大规模 ≤1.2x | kernel launch latency 在小数据量上占比高 | 正常，记录规模趋势 |
| 固定循环代价 | 所有规模 >4x，固定倍数 | 数据依赖 while → range(N) 固定迭代 | 记录，不推荐性能敏感场景 |
| view vs copy | 所有规模远大于 1x | PyTorch 是 view（零拷贝），kernel 是拷贝 | 先天劣势，记录为不支持场景 |
| 多次 launch | 随分片数线性恶化 | 每次 slice/chunk 一次 kernel launch | 优化方向：合并为单次 kernel |

**性能回退定位**（比率 > 1.5x 时）：

1. **检查 block_size**：auto-tune 是否选择合理值
2. **检查内存访问**：是否 coalesced access
3. **检查冗余 load/store**：是否有不必要的全局内存读写
4. **检查广播计算**：是否有不必要的广播开销
5. **检查 tile 配置**：tile 大小是否合理
6. **检查 launch 次数**：是否有多余的 kernel 调用（如逐 slice 调用）

### 阶段 6：输出报告

**报告模板**（每算子一份，存入 `reports/{op_name}_report.md`）：

```markdown
# {op_name} 算子开发报告

## 1. 算子信息
- 名称、分类、共享/自定义 arrangement、关键 DSL 操作、基线、生成文件

## 2. 精度验证
- 表格：每个测试用例的名称、dtype、规模、PASSED/FAILED
- 四项必检结果

## 3. 性能评估
- 表格：至少 3 种规模 × 至少 1 种 dtype
- 六项策略逐项评估（含"不适用"的原因）
- 性能结论：是否达标、根因分析

## 4. 边界情况
- 已处理的特殊场景列表
- 已知限制

## 5. 迭代历史
- 每次迭代：现象 → 根因 → 修复 → 验证结果

## 6. 合计
- 总迭代次数、精度通过率、性能目标达成情况
```

---

## 四、终止条件

| 条件 | 处理 |
|------|------|
| 连续 3 次精度验证不通过 | 停止并报告精度问题 |
| 连续 3 次性能改进幅度 < 5% | 接受当前最优结果 |
| 总迭代次数达到 10 次 | 输出当前结果 |

迭代计数器：
- `accuracy_failures`：连续精度失败次数（成功时重置）
- `performance_improvements`：连续 <5% 性能改进次数
- `total_iterations`：总迭代次数（上限 10）

---

## 五、静态验证清单

在运行测试之前，对生成的代码做以下检查：

- [ ] kernel 文件只 import 允许的模块（functools, enum, math, copy, ninetoothed, ninetoothed.language）
- [ ] 用到 `ninetoothed.float64` 等常量时，确保有 `import ninetoothed`（不能只有 `import ninetoothed.language`）
- [ ] application 函数体内没有 `torch.*`、`triton.*`、`cuda.*`
- [ ] application 函数体内没有引用外部变量或闭包变量
- [ ] premake 函数存在且返回 `(arrangement_, application, tensors)` 三元组
- [ ] 所有 Tensor 声明正确指定了 ndim
- [ ] 标量参数使用正确方式：constexpr 或同类型 Tensor
- [ ] 需要的 padding fill 值通过 `other=` 设置
- [ ] premake 中使用 `functools.partial` 绑定 arrangement 参数
- [ ] application 中输出赋值有 `# noqa: F841`
- [ ] torch 层使用 `_cached_make` 编译 kernel
- [ ] torch 层正确处理了 None 默认值
- [ ] torch 层的输出张量有正确的 shape/dtype/device
- [ ] torch 层传给 kernel 的参数数量 = premake 中 Tensor 的数量（包括 constexpr Tensor）
- [ ] 标量系数需要与不同 dtype 输入做运算时，优先用 constexpr 而非 `Tensor(0, dtype=ninetoothed.float64)`

---

## 六、错误恢复策略

| 错误类型 | 症状 | 诊断 | 修复方向 |
|----------|------|------|----------|
| Import 错误 | ModuleNotFoundError | kernel 中用了禁止的 import | 改用 ntl 等价函数 |
| 编译错误 | Triton 编译失败 | arrangement 链中 tile/expand/squeeze 不匹配 | 检查 tile 形状和 squeeze 维度 |
| NaN | isnan 断言失败 | 除零、log 负数、sqrt 负数 | 加 epsilon，用 ntl.where 保护 |
| Inf | isinf 断言失败 | exp 溢出（尤其 float16） | 转 float32 计算，用 exp2 trick |
| 全部错误值 | allclose 差异很大 | arrangement 映射错误 | 检查 permute/tile/flatten 顺序 |
| 边界错误 | 只有边界值错 | 缺少 padding fill 或 other 值不对 | 设置正确的 other 参数 |
| 形状不匹配 | shape mismatch | 输出分配或 arrangement 链错误 | 检查 torch 层的输出 shape 计算 |
| 运行慢 | kernel 耗时长 | block_size 不优或内存访问差 | 调整 block_size 范围，尝试不同 tile 顺序 |
| NameError（编译时） | `'xxx' is not defined` | application 内引用了闭包/外部变量 | 改用 `Tensor(0, constexpr=True, value=...)` 传入 |
| NameError（premake） | `'ninetoothed' is not defined` | kernel 文件缺少 import | 确认 `import ninetoothed` 存在 |
| 类型不兼容 | `IncompatibleTypeError` | 0-dim 标量 Tensor 的 dtype 与输入不匹配 | 标量系数改用 constexpr |
| 参数数量不匹配 | `takes N positional arguments but M were given` | torch 层传参数量不一致 | 检查 premake 中 Tensor 数量 = kernel 调用参数数量 |

---

## 七、测试生成协议

为每个算子生成测试文件：`ntops/tests/test_{op_name}.py`

```python
import os
import pytest
import torch
import ntops

DTYPE_TOLERANCES = [
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-3, 1e-3),
]

@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_{op_name}_basic(dtype, rtol, atol):
    # ... 基本测试 ...

@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_{op_name}_large(dtype, rtol, atol):
    # ... 大尺寸测试 ...

def test_{op_name}_edge_cases():
    # ... 边界情况测试 ...
```

运行：`cd $PROJECT_ROOT/ntops && python -m pytest tests/test_{op_name}.py -v`

---

## 八、外部调用能力

| 能力 | 调用方式 | 用途 |
|------|----------|------|
| `_cached_make` | `from ntops.torch.utils import _cached_make` | 编译并缓存 GPU kernel |
| `ninetoothed.make()` | 通过 `_cached_make` 间接调用 | 将 arrangement+application 编译为 GPU kernel |
| `pytest` | `python -m pytest tests/test_{op}.py -v` | 运行精度验证测试 |
| `torch.allclose` | 在测试脚本中使用 | 精度比对 |

**已验证行为**：NineToothed 自动处理 stride 信息，element-wise（relu, silu, leaky_relu）、binary（add）、reduction（softmax）算子均可直接接受转置张量和步幅切片，无需 torch 层做 `.contiguous()` 转换。

---

## 九、Generated Source 检查与性能分析

### Generated Source 检查（JIT 模式）

当 `caller="torch"`（默认 `_cached_make`）时，NineToothed 将生成的 Triton 源码缓存到
`~/.ninetoothed/{sha256_hash}.py`。

**获取生成源码路径**：

```python
import ninetoothed
from ntops.kernels.leaky_relu import premake

# 获取编译后的 kernel handle
kernel = ninetoothed.make(*premake(2), caller="torch")

# kernel._source 是生成源文件的路径
print(kernel._source)  # ~/.ninetoothed/abc123....py
```

**读取并检查生成源码**：

```python
with open(kernel._source) as f:
    source = f.read()
print(source)
```

**检查清单（逐项对照生成源码）**：

1. **tile 映射** — 搜索 `triton.language.load` / `triton.language.store`，确认指针偏移量与 arrangement 链中的 tile/expand/squeeze 一致
2. **数据类型** — 搜索 `.to(` 或 `cast`，确认累加器使用了 `float32`，输出前转回了原始 dtype
3. **内存访问模式** — 检查 `triton.language.load` 的 `mask` 参数，确认 padding 边界处理；检查是否有冗余的 load/store
4. **constexpr** — 搜索 `ninetoothed_constexpr`，确认编译时常量作为函数参数正确传入
5. **block_size** — 搜索 `BLOCK_SIZE`，确认 auto-tune 的 `num_warps`/`num_stages` 配置合理（warps=8, stages=3 为默认值）

**常见问题与修复**：

| 生成源码中的症状 | 根因 | 修复 |
|----------|------|------|
| 出现 `float64` 类型 | 某处用了 `ninetoothed.float64` | 改为 `ntl.float32` 或 constexpr |
| load 无 mask 但 tile 大小不等于输入大小 | 缺少 padding 保护 | 检查 other 参数，确认 boundary check |
| 多次 store 到同一位置 | arrangement 重复映射 | 检查 tile/expand/squeeze 链 |
| 冗余的 `load` + `store`（identity） | application 做了无意义的拷贝 | 确认 arrangement 已经完成了所需的数据移动 |

### AOT Build（`caller="cuda"` 模式）

AOT 编译在运行前将 kernel 编译为 `.so` 共享库，避免运行时 JIT 编译开销。
适用于部署场景或需要分发预编译 kernel 的情况。

**基本 AOT 编译**：

```python
import ninetoothed
from ntops.kernels.relu import premake

# caller="cuda" 触发 AOT 编译
kernel = ninetoothed.make(
    *premake(2),
    caller="cuda",
    kernel_name="relu",
    output_dir="./build",
    num_warps=4,
    num_stages=2,
)
```

**输出文件**（写入 `output_dir`）：

| 文件 | 内容 |
|------|------|
| `{name}.h` | C 头文件，声明 `launch_{name}()` |
| `{name}.cpp` | C++ 调度器，运行时选择 kernel 变体 |
| `{name}.{hash}.cpp` | 每个变体的 CUDA C++ wrapper（含嵌入的 PTX） |
| `{name}.so` | 编译后的共享库 |

**多配置批量 AOT（`ninetoothed.build()`）**：

```python
import ninetoothed
from ntops.kernels.relu import premake

configs = [
    ((2,), {"dtype": "float32"}, {"num_warps": 4}),
    ((2,), {"dtype": "float16"}, {"num_warps": 8}),
]

ninetoothed.build(
    premake,
    configs,
    caller="cuda",
    kernel_name="relu",
    output_dir="./build",
    lazy=False,  # True = 延迟编译到首次调用，避免导入死锁
)
```

**关键参数**：

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `num_warps` | 每个 block 的 warp 数 | 4/8（越大占用资源越多，可能提升吞吐） |
| `num_stages` | 软件流水线 stage 数 | 2（默认），增大可隐藏延迟 |
| `lazy` | 延迟编译 | `True` 用于避免导入时编译死锁 |
| `max_num_configs` | auto-tune 搜索的最大配置数 | 默认自动，可手动限制加速 debug |

**JIT vs AOT 选择决策**：

| 场景 | 推荐模式 |
|------|----------|
| 开发调试阶段 | JIT（`caller="torch"`） |
| 生产部署 / 预分发 | AOT（`caller="cuda"`） |
| 需要检查 generated source | JIT（`.py` 文件可直接阅读） |
| 多 dtype/shape 组合 | `build()` 批处理 AOT |

### 布局调试工具

当精度验证失败或怀疑 arrangement 映射错误时，使用以下工具：

**`simulate_arrangement`** — 跟踪元素索引：

```python
from ninetoothed.debugging import simulate_arrangement
from ntops.kernels.meshgrid import arrangement, premake

arrangement_, _, tensors = premake(2)
source_tensors, target_tensors = simulate_arrangement(arrangement_, tensors)

# source_tensors[i] — 输入张量的索引映射（-1 = padding）
# target_tensors[i] — 输出张量的索引映射
print(target_tensors[0])  # 每个目标元素来自输入的哪个位置
```

**`ninetoothed.eval()`** — 评估单个 Tensor 的偏移映射：

```python
import ninetoothed
from ninetoothed import Tensor

t = Tensor(2, dtype="float32").tile((64, 64))
offsets = ninetoothed.eval(t)
print(offsets)  # numpy 数组，-1 表示 padding 位置
```

### Benchmark 设计规范

为性能敏感算子设计 benchmark 时：

1. **基线选择**：与 PyTorch 同功能 API 对比
2. **输入规模**：至少覆盖小（256x256）、中（1024x1024）、大（4096x4096）三种
3. **warmup**：至少 10 次预热迭代
4. **repeat**：至少 100 次测量取平均
5. **同步**：使用 `torch.cuda.synchronize()` 确保计时准确
6. **结果报告**：包含 ntops 耗时、PyTorch 耗时、比率

### 性能回退定位

当性能明显落后时（比率 > 1.5x），按以下步骤诊断：

1. **检查 block_size**：auto-tune 是否选择了合理的值
2. **检查内存访问**：是否 coalesced access
3. **检查冗余 load/store**：是否有不必要的全局内存读写
4. **检查广播计算**：是否有不必要的广播开销
5. **检查 tile 配置**：tile 大小是否合理
6. **检查 contiguous/stride**：非连续输入是否有额外开销

### 非连续输入开销分析

NineToothed 自动处理 stride 信息，但非连续输入可能有性能影响：
- 转置张量：通常无显著开销（stride 由 Triton 处理）
- 步幅切片：取决于 stride 模式，极端情况可能有影响
- 建议：对性能敏感场景，实测对比连续 vs 非连续输入

---

## 十、失败诊断与记录协议

### 记录格式

每次失败必须记录以下信息：

```
### 失败 #N
- **现象**：[具体错误信息]
- **诊断路径**：[如何定位到根因]
- **根因判断**：[根本原因]
- **最小修复**：[具体代码改动]
- **验证命令**：`python -m pytest ...`
- **验证结果**：PASSED / FAILED
```

### 诊断流程

```
失败现象
├── 编译错误 → 检查 import / arrangement 链 / Tensor 声明
├── 运行时错误 → 检查参数数量 / shape 分配 / dtype 兼容性
├── 精度错误 → 检查累加精度 / padding 值 / cast 位置
└── 性能问题 → 检查 block_size / 内存访问 / 冗余计算
```

### 修复原则

1. **最小修复**：只改必要部分，不做无关重构
2. **不修改 arrangement**：优先修改 application 或 torch 层
3. **保持风格一致**：遵循现有代码的命名和结构风格
4. **验证闭环**：每次修复后立即运行测试确认

### 迭代协议

当算子验证失败需要迭代时，按以下步骤执行：

1. 读取错误信息或测试输出
2. 分类错误：编译 / 精度 / 性能
3. 确定根因（参考错误恢复策略表）
4. 给出具体修复指令：
   - 错误信息（原样）
   - 根因分析
   - 具体修复要求（如"在 exp 之前加 float32 类型转换"）
   - 约束（如"不要修改 arrangement，只修改 application"）
5. 重新运行验证
6. 更新迭代计数器

---

## 十一、不支持场景说明

当遇到以下场景时，需明确说明不支持：

| 场景 | 原因 | 建议 |
|------|------|------|
| 动态 shape（运行时变化维度数） | NineToothed 的 premake 需要 ndim 参数 | 在 premake 中固定 ndim |
| 不同 ndim 的广播输入（如 `(4,1,256)` × `(1,256)`） | `element_wise` arrangement 要求所有输入同 ndim 或 0-dim scalar | 在 torch 层用 `unsqueeze`/`expand` 统一 ndim 后再传入 kernel |
| 特定 dtype 不兼容 | 如 float64 与某些 GPU 操作 | 在文档中注明支持的 dtype |
| 递归算法 | GPU kernel 禁止递归 | 改用迭代实现 |
| 数据依赖循环（while b!=0） | Triton 不支持 | 改用 range 固定循环。**关键陷阱**：循环体内所有状态变量更新必须条件化 — 一旦算法收敛，所有赋值必须是 no-op（`x = ntl.where(converged, x, new_x)`），详见 pitfalls #15 |
| 大规模 matmul（Windows） | Windows triton 优化不足 | Linux 环境下性能更好 |
| 动态索引访问（`tile[j]`） | Triton JIT 不支持按动态变量索引 tile 内元素 | 改用向量操作（`ntl.where(tile == scalar, ...)`），或使用 `ninetoothed.language.gather(src, indices)` 按索引收集 |
| **递归算法** | GPU kernel 禁止递归 | 改写为迭代实现 |
| 大规模 matmul（Windows） | Windows triton 优化不足 | Linux 环境下性能更好 |

> **注意**：以下能力已通过 `ninetoothed.language.xxx` 直接引用验证通过（参考 §二 Application 编写关键模式 #7）：
> - `ninetoothed.language.histogram`：整数直方图（需 int32 输入，`num_bins` 参数）
> - `ninetoothed.language.atomic_add`：原子加（需 pointer）
> - `ninetoothed.language.gather`：按索引收集元素
> 这些 Triton 原语通过 code generator 的自动转换（`ninetoothed.language.xxx` → `triton.language.xxx`）可用，绕过了 `ntl` 模块的限制。**注意** `ntl.xxx` 不可用（会报 AttributeError），必须用全限定名 `ninetoothed.language.xxx`。

---

## 十二、详细参考资料索引

本 Skill 包含以下参考资料，按需查阅：

| 文件 | 内容 |
|------|------|
| `references/code_templates.md` | 9 种 Arrangement 模式的完整代码模板 |
| `references/ntl_api.md` | ntl（ninetoothed.language）API 参考 |
| `references/tensor_guide.md` | Tensor 声明与元操作参考 |
| `references/pitfalls.md` | 14 种常见陷阱及修复方法 |
| `examples/01_elementwise_leaky_relu.md` | 完整 leaky_relu 开发示例（含 5 次迭代修复） |
| `examples/02_reduction_log_softmax.md` | 完整 log_softmax 开发示例（数值稳定） |
