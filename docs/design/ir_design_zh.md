# 九齿多后端 IR 设计文档

日期：2026-06-13

## 1. 背景与目标

九齿目前以 Python DSL 描述张量算子，既要保留已有 Triton 后端的生产路径，又要扩展到 CUDA、TileLang、TVM，以及后续可能出现的 AscendC、BangC 等设备编程语言。不同后端的语法、内存层级、并行模型和优化接口差异很大，如果直接从九齿 DSL 分别生成每一种目标代码，会很快形成多个相互漂移的代码生成器，难以验证语义一致性，也难以复用优化。

因此，九齿需要一套多层 IR。它的核心目标不是简单把 Triton 代码翻译成其他语言，而是把九齿 DSL 中的张量语义、索引语义、归约语义、调度语义和目标语言语义分层表达：

- 前端 DSL 只负责表达用户意图和张量布局约束。
- 语义 IR 描述“这个算子计算什么”。
- 调度 IR 描述“这个算子如何映射到硬件并行结构”。
- 目标 IR 描述“如何用某个后端语言合法、可编译、可运行地表达”。
- 验证体系证明不同后端实现与同一语义 IR 等价。

当前仓库已经落地了第一阶段的紧凑 IR，包括 `TensorTypeIR`、`ExprIR`、`ProgramIR`、`KernelIR` 和统一的 `BackendArtifact`。这套 IR 已经可以作为多后端生成的共同入口，并支撑 Triton、CUDA、TileLang、TVM 四种后端入口，其中 CUDA、TileLang、TVM 对结构化 `ProgramIR` 已具备原生代码生成能力。

本文档给出完整的 IR 设计，说明它的理论语义、可表达性、可扩展性、当前能力满足情况，以及后续演进方向。

## 0.1 当前新增：SSAProgramIR 语义层

2026-06-14 起，九齿新增了第一版 SSA-like IR：

- `SSATypeIR`：描述 SSA value 的类别、dtype、shape 和扩展属性。
- `SSAValueIR`：描述具名 SSA value，例如输入张量 `x` 或临时值 `%0`。
- `SSAOperationIR`：描述单条 SSA op，包括 opcode、operands、results、attrs 和可选 regions。
- `SSABlockIR`：描述直线 SSA block。
- `SSAProgramIR`：描述从 `ProgramIR` 规范化得到的语义 SSA 程序。
- `program_to_ssa(program, tensors)`：把现有结构化 `ProgramIR` 转换为 `SSAProgramIR`。

这层 IR 目前是 semantic SSA，不直接表达线程、tile、shared memory 等调度信息。它的职责是先把嵌套 `ExprIR` 树展开成显式数据流：

```text
%0 = reduce.sum x {axis = 1}
%1 = arith.div %0, 32.0
%2 = arith.mul x, x
%3 = reduce.sum %2 {axis = 1}
%4 = arith.div %3, 32.0
%5 = arith.mul %1, %1
%6 = arith.sub %4, %5
%7 = arith.add %6, 1.0e-5
%8 = math.rsqrt %7
%9 = arith.sub x, %1
%10 = arith.mul %9, %8
%11 = arith.mul %10, weight
%12 = arith.add %11, bias
mem.store %12, out
```

第一版 SSA 已经覆盖：

- elementwise 表达式
- fill/copy
- 1D reduction
- axis reduction
- rowwise softmax/layernorm/rmsnorm 这类带行规约和广播的融合表达式
- transpose
- matmul
- flash attention

它与当前后端的关系是渐进式的：`KernelIR` 同时携带 `program: ProgramIR` 和 `ssa: SSAProgramIR`。现有 TileLang、TVM 后端仍以 `ProgramIR` 为主输入，因此不会破坏已经通过的多后端生成；CUDA 后端已经开始迁移，线性 elementwise/fill/copy/multi-output/offsets 这类 SSA block 可以直接由 `SSAProgramIR` 渲染为 CUDA C++，不再从嵌套 `ExprIR` 重新遍历。新优化 pass 和后续后端迁移可以继续读取 `SSAProgramIR`。

当前 SSA 化还包含一个局部纯表达式 CSE：相同的 `ExprIR` 子树会复用同一个 SSA value。例如 layernorm 中重复出现的 `sum(x, axis=1)` 只生成一次 `reduce.sum x`，后续表达式复用 `%0`。这不是完整全局优化器，但已经把九齿从“表达式树摘要”推进到了可承载优化 pass 的数据流 IR。

## 2. 设计原则

### 2.1 语义和实现分离

IR 首先要表达数学语义，而不是某个目标语言的语法。比如 row-wise softmax 的本质是：

1. 对每一行求最大值。
2. 对每个元素计算指数。
3. 对每一行求指数和。
4. 做逐元素归一化。

它可以被实现为 Triton block program、CUDA kernel、TileLang program 或 TVM TensorIR block。后端的线程绑定、block 大小、共享内存和向量化都不应污染语义 IR。

### 2.2 多层而不是单层

单层 IR 要么太高层，无法表达底层语言的优化细节；要么太低层，失去从九齿 DSL 直接推导和验证的便利。九齿应采用如下多层结构：

| 层次 | 职责 | 当前状态 |
| --- | --- | --- |
| Tensor/Shape IR | 参数、dtype、rank、shape、jagged、constexpr、布局事实 | 已有 `TensorTypeIR` 雏形 |
| Expr IR | 标量表达式、张量索引、数学函数、条件表达式 | 已有 `ExprIR` 雏形 |
| Program IR | 算子的语义操作序列，如 elementwise、reduction、rowwise、matmul、transpose | 已有 `ProgramIR` 和若干 op |
| Schedule IR | tile、split、reorder、thread binding、warp mapping、vectorize、unroll、pipeline | 设计阶段 |
| Memory/Layout IR | buffer、memory scope、layout map、mask、shared/local/register 生命周期 | 设计阶段 |
| Target IR | Triton/CUDA/TileLang/TVM/AscendC/BangC 的目标语言节点与 ABI | 部分后端已渲染 |

### 2.3 可验证优先

IR 的每一个层次都应有清晰的等价关系：

- 前端 DSL 到 Program IR：保持用户算子的数学语义。
- Program IR 到 Schedule IR：保持计算结果不变，只改变执行顺序和并行映射。
- Schedule/Memory IR 到 Target IR：保持语义和内存可见性约束。
- Target IR 到运行结果：通过编译、运行和数值对比验证。

这也是当前实现采用 `BackendArtifact` 的原因：生成物必须能被保存、审计、编译、运行，并与 PyTorch/NumPy 参考结果对比。

## 3. 理论模型

### 3.1 张量语义

设符号形状环境为：

```text
Γ = { n0: Nat, n1: Nat, ..., dtype facts, layout facts }
```

一个 rank 为 `r` 的张量 `A` 可以被看作从有限整数域到标量域的函数：

```text
A : D_A -> τ
D_A = [0, n0) x [0, n1) x ... x [0, n(r-1))
```

其中 `τ` 是 dtype 对应的标量类型，如 `float32`、`float16`、`int64`、`bool`。布局不改变张量的数学域，只改变逻辑索引到物理地址的映射：

```text
addr_A : D_A -> Ptr
```

因此，Program IR 中的算子首先在逻辑索引空间上定义，后续 Memory/Layout IR 再把逻辑索引 lowering 到物理地址。

### 3.2 表达式语义

`ExprIR` 是一个带类型的表达式代数。给定输入张量环境 `ρ`、符号形状环境 `Γ` 和当前索引环境 `I`，表达式有解释函数：

```text
[[e]](Γ, ρ, I) -> scalar
```

典型表达式包括：

- 常量：`const(c)`
- 变量：`var(x)`
- 一元运算：`unary(op, e)`
- 二元运算：`binary(op, lhs, rhs)`
- 数学函数：`call(fn, args...)`
- 条件选择：`where(cond, t, f)`
- 索引读取：`load(tensor, logical_index)`
- 归约引用：`reduce(op, axis, body)`

当前实现中的 `ExprIR(kind, value, args)` 是一个紧凑表示，已经覆盖了大量一元、二元和数学函数调用。完整形态中应进一步把 `kind` 从字符串提升为显式节点类型，以便做类型检查、模式匹配和优化。

### 3.3 Program IR 的指称语义

`ProgramIR` 描述一个有限操作序列。每个 op 都有输入张量、输出张量、逻辑迭代域和表达式主体。最基础的 elementwise assignment 可以写作：

```text
for i in Ω:
    Y[i] = [[e]](Γ, ρ, {i})
```

其中 `Ω` 是输出张量的逻辑域。归约 op 可以写作：

```text
for r in Rows:
    Y[r] = reduce_{c in Cols}(op, [[e]](Γ, ρ, {row: r, col: c}))
```

row-wise broadcast reduction 可以写作：

```text
tmp[r] = reduce_{c in Cols}(op, body(r, c))
for c in Cols:
    Y[r, c] = f(X[r, c], tmp[r], ...)
```

这说明 row-wise softmax、layernorm、rms_norm 等算子不是特殊语法，而是“行内归约 + 逐元素广播写回”的组合。

### 3.4 等价关系

两个后端生成物 `P_backend1` 和 `P_backend2` 对同一个 `ProgramIR P` 是等价的，当且仅当对任意合法输入 `x`，它们在允许的浮点误差范围内产生相同输出：

```text
∀x ∈ LegalInputs(P):
    run(P_backend1, x) ≈ run(P_backend2, x) ≈ [[P]](x)
```

这里的 `≈` 不是简单逐 bit 相等。对浮点算子，等价关系应由 dtype、归约顺序、目标后端 math intrinsic 精度共同确定，例如 `rtol/atol` 或 ULP 范围。当前验证采用 PyTorch/NumPy 参考结果加数值容差对比，这是后续 property-based testing 和 differential testing 的基础。

## 4. 当前 IR 结构

当前实现位于 `src/ninetoothed/ir.py`，是完整多层设计的第一阶段切片。

### 4.1 `TensorTypeIR`

`TensorTypeIR` 是公开 kernel 参数的后端无关类型事实：

- `name`：参数名。
- `ndim`：rank。
- `dtype`：标量类型。
- `shape`：符号或静态 shape。
- `constexpr`：是否为编译期常量。
- `jagged_dim`：jagged tensor 的特殊维度。

它提供了最小但稳定的 ABI 信息，使 CUDA、TileLang、TVM 后端可以共享参数列表和 shape 推断。

### 4.2 `ExprIR`

`ExprIR` 当前是紧凑表达式树：

```python
ExprIR(kind: str, value: Any = None, args: tuple[ExprIR, ...] = ())
```

它已经可以表达：

- 变量和常量。
- 一元和二元标量运算。
- 常见数学函数，如 `exp`、`log`、`sqrt`、`rsqrt`、`tanh`、`erf` 等。
- `where` 条件选择。
- row-wise lowering 中的归约子表达式。

它的优势是简单、可序列化、便于快速扩展。它的不足是类型和 effect 尚未成为显式系统，后续应演进为一组 dataclass 节点或 algebraic data type。

### 4.3 `ProgramIR`

`ProgramIR` 是当前最关键的语义层。它把九齿 application AST 中可结构化的模式转换为后端无关的 op。目前包括：

| IR 节点 | 语义 |
| --- | --- |
| `ElementwiseAssignOpIR` | `output[i] = expression(i)`，支持 multi-output |
| `AxisReductionAssignOpIR` | 二维 row-wise reduction，输出为行向量 |
| `RowwiseAssignOpIR` | 二维 row-wise reduction 后广播写回 |
| `FillOpIR` | 填充常量 |
| `CopyOpIR` | 张量复制 |
| `ReductionOpIR` | 一维归约 |
| `MatmulOpIR` | 基础二维矩阵乘 |
| `TransposeOpIR` | 二维转置 |
| `ElementwiseBinaryOpIR` | 早期二元逐元素 op 兼容层 |

`ProgramIR.kind` 当前仍是字符串标签，如 `elementwise`、`reduction`、`axis_reduction`、`rowwise`、`matmul`、`transpose`。后续可以把 kind 作为派生属性，避免字符串与 op 类型之间出现不一致。

### 4.4 `KernelIR`

`KernelIR` 是 kernel 级记录，承担前端 lowering 和后端 registry 之间的契约：

- `kernel_name`
- `source`
- `source_language`
- `entrypoint`
- `launch`
- `tensors`
- `compiler_options`
- `metadata`
- `program`

它允许两种路径共存：

1. 旧 Triton 生成器先生成 Triton source，再由 `KernelIR.from_codegen` 包装成 artifact。
2. 新结构化路径直接从九齿 application AST 推断 `ProgramIR`，然后由 CUDA、TileLang、TVM 原生 lowerer 渲染目标代码。

这个设计使九齿可以渐进迁移，而不需要一次性重写成熟的 Triton 生成路径。

## 5. 可表达性分析

### 5.1 已覆盖的表达能力

当前 IR 对以下算子族具备实际表达和生成能力：

- 逐元素表达式：加减乘除、比较、逻辑、常见数学函数、`where`。
- 多输出逐元素：一个 application 产生多个输出 assignment。
- 一维归约：`sum`、`max`、`min` 等基础归约形式。
- 二维按行归约：`axis=1` 的 `sum/max/min`。
- row-wise broadcast reduction：softmax、log_softmax、layernorm、rms_norm、skip/add norm 类模式。
- 基础结构化算子：fill、copy、transpose、matmul。
- shape-aware indexing 和偏移表达式的部分模式。

从理论上看，这些算子都属于“有限张量域上的组合表达式 + 有界归约”范畴。只要表达式语言对标量运算封闭，并且 Program IR 允许在有限索引域上绑定变量，就可以表达大量深度学习基础算子。

### 5.2 对九齿当前能力的满足性

当前实现已经满足以下能力：

- `ninetoothed.lower(..., backend="triton")`：保留现有 Triton 路径和 artifact 输出。
- `ninetoothed.lower(..., backend="cuda")`：对结构化 `ProgramIR` 生成 CUDA C++ 源码。
- `ninetoothed.lower(..., backend="tilelang")`：对结构化 `ProgramIR` 生成 TileLang Python 源码。
- `ninetoothed.lower(..., backend="tvm")`：对结构化 `ProgramIR` 生成 TVMScript/TensorIR 源码。
- 统一 `BackendRegistry`、`BackendOptions`、`BackendCapability`、`BackendArtifact`。
- 生成物可以写入文件，作为后续编译、运行和审计对象。

已有验证结果显示：

| 验证项 | 结果 |
| --- | --- |
| ninetoothed 完整 pytest | `262 passed, 1 skipped, 70 subtests passed` |
| 后端/IR/AST inference 单元测试 | `47 passed` |
| 真实九齿 DSL 自动 lowering | `24 case x 3 backend = 72` 项通过 |
| 表达式 stress 验证 | `120 case x 3 backend = 360` 项通过 |
| 结构化算子验证 | `60 case x 3 backend = 180` 项通过 |
| 可执行验证总量 | `204` 个 case，三后端合计 `612` 个验证项 |
| ntops.lab ProgramIR 审计 | 246 个算子中 215 个可识别，14 个不可识别，17 个缺少 application |
| ntops.lab 三后端 lowerability | CUDA/TileLang/TVM 各 215 个可 lower，0 个 partial failure |

这说明当前 IR 已经能支撑第一阶段多后端代码生成闭环：同一个九齿语义片段可以自动生成 CUDA、TileLang、TVM 代码，并通过编译运行和数值对比验证。

### 5.3 当前不能完整表达的能力

当前 IR 仍有明确边界。ntops.lab 中剩余 14 个不可识别 application 主要落在两类：

- matmul/linear epilogue 模式：`acc = zeros; for k: acc += dot(...); out = epilogue(acc, bias/c)`。
- 显式循环索引模式：如 `cumsum`、`upsample_linear1d` 中的 loop-carried dependency、masked store、相邻元素 stencil。

这些不是后端渲染问题，而是 Program IR 尚未把对应语义建模出来。后续应新增：

- `DotLoopOpIR` 或 `MatmulEpilogueOpIR`
- `ScanOpIR`
- `WindowOpIR` / `StencilOpIR`
- `MaskedStoreOpIR`
- 更显式的 `LoopIR`

引入这些节点后，CUDA、TileLang、TVM 后端可以复用同一语义层继续扩展，而不是分别在每个后端里识别 Python AST。

## 6. 可扩展性设计

### 6.1 新语义节点扩展

新增算子族时，应优先判断它是否是已有节点的组合。如果不能自然表达，再新增 Program IR 节点。推荐扩展顺序如下：

| 新节点 | 目标覆盖 |
| --- | --- |
| `MatmulEpilogueOpIR` | `mm`、`bmm`、`addmm`、`gemm_bias`、`linear`、GELU/ReLU epilogue |
| `LoopIR` | 有界 for loop、循环内局部变量、循环不变量 |
| `ScanOpIR` | `cumsum`、prefix sum、递推状态 |
| `WindowOpIR` / `StencilOpIR` | upsample、conv-like local neighborhood |
| `GatherOpIR` / `ScatterOpIR` | index_select、embedding、scatter、segment 类算子 |
| `AtomicOpIR` | scatter-add、histogram、并发更新 |
| `RandomOpIR` | dropout、采样类算子，需要显式 RNG state |

每个新节点都应定义：

- 输入和输出张量。
- 逻辑迭代域。
- 读写 effect。
- 类型和 shape 推导规则。
- 合法性约束。
- 参考解释器或 Python reference。
- 各后端 lowering 义务。

### 6.2 Schedule IR 扩展

当前 CUDA、TileLang、TVM 的生成偏 correctness-first。性能优化需要独立的 Schedule IR，而不是把调度写死在 Program IR 中。建议 Schedule IR 包含：

- `Tile(axis, factors)`
- `Split(axis, outer, inner)`
- `Reorder(axes)`
- `Bind(axis, target)`，如 `blockIdx.x`、`threadIdx.x`、`warp`。
- `Vectorize(axis, width)`
- `Unroll(axis, factor)`
- `Pipeline(stage_count)`
- `CacheRead(tensor, scope)`
- `CacheWrite(tensor, scope)`
- `ComputeAt(block, axis)`
- `Inline(expr/block)`

这样同一个 `MatmulEpilogueOpIR` 可以生成：

- Triton block-level program。
- CUDA shared-memory tiled kernel。
- TileLang tile program。
- TVM TensorIR schedule。

### 6.3 Memory/Layout IR 扩展

底层语言差异往往集中在内存和布局上。建议引入：

- `BufferIR`：buffer 名称、dtype、shape、scope、alignment。
- `LayoutMapIR`：逻辑 index 到物理 offset 的仿射或半仿射映射。
- `MaskIR`：边界保护、jagged 有效性、predicate。
- `MemoryScopeIR`：global、shared、local、register、tensor core fragment。
- `BarrierIR`：线程同步、memory fence。
- `AsyncCopyIR`：CUDA `cp.async`、TMA，或目标后端对应机制。

这会让后续 AscendC、BangC 等国产卡后端只需要描述自己的 memory hierarchy 和 intrinsic，而不是重新解释九齿 DSL。

### 6.4 Target Capability 扩展

每个目标后端应声明 capability，而不是让 lowering 过程隐式失败。Capability 至少包括：

- 支持的 dtype。
- 支持的 math intrinsic。
- 最大 block/thread 限制。
- 支持的 memory scope。
- 是否支持 tensor core 或矩阵专用指令。
- 是否支持动态 shape。
- 是否支持 atomic。
- 是否支持 warp-level primitive。
- 编译器和 runtime 依赖。

合法性检查可以写作：

```text
legal(P, S, Target) -> bool
```

其中 `P` 是 Program IR，`S` 是 Schedule IR。只有合法组合才进入目标代码渲染。

## 7. Lowering 流程设计

推荐完整 lowering pipeline 如下：

```text
NineToothed DSL
    |
    v
Frontend AST / Tensor Arrangement Facts
    |
    v
Tensor + Shape IR
    |
    v
Program IR
    |
    +--> Type/Shape/Effect Verification
    |
    v
Canonicalization and Algebraic Simplification
    |
    v
Schedule IR Selection
    |
    v
Memory/Layout IR Realization
    |
    v
Target IR Rendering
    |
    v
BackendArtifact
    |
    v
Compile / Run / Compare
```

当前实现处于以下阶段：

- 已有 `lower()` 作为统一 API。
- 已有 AST inference，把可识别 application 转成 `ProgramIR`。
- 已有 CUDA、TileLang、TVM 后端渲染结构化 `ProgramIR`。
- Triton 仍主要通过原有生成器产生源代码，再包装为 `KernelIR`/`BackendArtifact`。
- Schedule IR 和 Memory/Layout IR 尚未独立成层。

这个状态是合理的第一阶段：先证明多后端语义闭环，再逐步把性能优化从后端渲染器中抽离出来。

## 8. 正确性与验证策略

### 8.1 单节点验证

每个 IR 节点都应有最小验证集合：

- 正常 shape。
- 边界 shape，如 1、非 2 的幂、不能整除 block size。
- 多 dtype。
- broadcasting 或 mask 边界。
- 随机输入和固定 seed。

### 8.2 后端差分验证

对同一 Program IR，应同时生成 CUDA、TileLang、TVM，必要时也生成 Triton，然后比较：

- 源码是否生成成功。
- 编译是否成功。
- 运行是否成功。
- 输出是否与 PyTorch/NumPy reference 一致。
- 不同后端之间是否一致。

### 8.3 ntops.lab 覆盖验证

ntops.lab 应作为真实算子覆盖集。建议分三层报告：

1. ProgramIR coverage：能否从真实九齿 application 自动识别 IR。
2. Backend lowerability：识别出的 ProgramIR 能否生成所有目标后端代码。
3. Executable correctness：生成代码能否编译运行并与参考结果比较。

当前已经完成第 1、2 层的大规模审计，并对选定结构化集合完成第 3 层可执行验证。下一步应把 remaining 14 个 unsupported application 纳入新 IR 节点设计，然后扩大第 3 层到全量 ntops.lab。

## 9. 对四后端的适配性

### 9.1 Triton

Triton 适合表达 block program、mask load/store、向量化表达式和 tile-level matmul。当前 Triton 是九齿原有生产路径。短期应继续把 Triton 作为 correctness baseline；中期应逐步支持从 Program/Schedule IR 重新渲染 Triton，使 Triton 和 CUDA/TileLang/TVM 使用同一语义来源。

### 9.2 CUDA

CUDA 后端需要显式表达 kernel 参数、grid/block、thread index、内存地址和同步。Program IR 到 CUDA 的关键是把逻辑迭代域映射到 `blockIdx/threadIdx`，并在 Memory/Layout IR 中处理边界和共享内存。当前实现已能生成可编译运行的 CUDA C++ baseline kernel，适合 correctness-first 验证。后续性能优化应通过 Schedule IR 添加 tiling、shared memory、warp-level reduction、tensor core intrinsic。

### 9.3 TileLang

TileLang 更接近 tile-level DSL，天然适合承接 Schedule IR。它可以把九齿的 Program IR 与 tile/block 结构清晰对应起来。当前实现已能生成 TileLang Python 模块并通过 TileLang 编译到 CUDA target。后续应让 Schedule IR 更直接地映射到 TileLang 的 tile primitives。

### 9.4 TVM

TVM TensorIR 有成熟的 block、axis、buffer、schedule 表达能力，适合作为九齿 IR 设计的外部参照。当前实现生成 TVMScript/TensorIR，并通过 `tvm.build(..., target="cuda")` 编译运行。后续可考虑把九齿 Schedule IR 映射到 TVM schedule primitive，以复用 TVM 的分析和优化能力。

## 10. 当前满足性结论

从“IR 是否足以支撑当前多后端能力”的角度，结论如下：

1. 对九齿已有测试和当前结构化算子集合，现有紧凑 IR 已经足够支撑 Triton、CUDA、TileLang、TVM 四后端入口。
2. 对 CUDA、TileLang、TVM，现有 `ProgramIR` 已能生成真实目标代码，并在已验证集合上完成编译、运行和数值对比。
3. 对 ntops.lab，当前 IR 可自动识别 215/246 个 application，识别出的 215 个在 CUDA、TileLang、TVM 三后端 lowerability 审计中全部成功。
4. 当前 IR 仍不能完整表达所有 ntops.lab 算子，特别是 matmul/linear epilogue、scan、stencil、masked loop-carried update。这需要新增 Program IR 节点，而不是简单补后端模板。
5. 从架构上看，当前 IR 已经满足第一阶段“统一语义入口 + 多后端 artifact + 可验证闭环”的要求；从长期高性能编译器角度看，还需要补齐 Schedule IR、Memory/Layout IR 和 Target Capability。

## 11. 后续演进路线

建议按以下顺序推进：

1. 引入显式类型系统和 verifier：检查 dtype、shape、rank、broadcast、reduction axis、输出写入唯一性。
2. 新增 `MatmulEpilogueOpIR`，优先覆盖 ntops.lab 剩余 linear 类算子。
3. 新增 `LoopIR`、`ScanOpIR`、`MaskedStoreOpIR`，覆盖 `cumsum`。
4. 新增 `WindowOpIR` 或 `StencilOpIR`，覆盖 `upsample_linear1d` 和后续局部邻域算子。
5. 引入 Schedule IR，把 correctness-first kernel 升级为可优化 kernel。
6. 引入 Memory/Layout IR，统一处理 shared/local/register、mask、jagged、layout。
7. 让 Triton 也从 Program/Schedule IR 渲染，完成四后端语义完全统一。
8. 为 AscendC、BangC 添加 Target Capability 和最小 Target IR renderer。
9. 建立全量 ntops.lab executable correctness gate。
10. 在正确性稳定后引入 cost model 和 autotuning。

## 12. 附录：当前实现文件

关键实现文件如下：

- `src/ninetoothed/ir.py`：IR dataclass 定义。
- `src/ninetoothed/lowering.py`：统一 lowering 入口和 AST 到 ProgramIR 的推断。
- `src/ninetoothed/backends/base.py`：后端 registry、capability、artifact 契约。
- `src/ninetoothed/backends/cuda.py`：CUDA codegen。
- `src/ninetoothed/backends/tilelang.py`：TileLang codegen。
- `src/ninetoothed/backends/tvm.py`：TVMScript/TensorIR codegen。
- `tests/test_backend_registry.py`：后端注册与 artifact API 测试。
- `tests/test_lowering_inference.py`：ProgramIR 推断测试。
- `tests/test_ssa_first_backend_lowering.py`：public `ninetoothed.lower()` 从 application AST 到 SSA 再到四后端 artifact 的端到端测试。
- `tests/test_ssa_pass_pipeline.py`：硬件无关与后端相关 pass pipeline、registry 和 metadata 测试。
- `tests/test_ssa_application_lowering.py`：application AST 到 SSA 的语义覆盖测试。
- `tests/test_kernel_ir.py`：`KernelIR`、`TensorTypeIR` 和 SSA 字段的结构契约测试。
