# 九齿 SSA IR Lowering Pipeline 设计

## 目标

九齿的核心 IR 不应把一个完整算子表示成一个粗粒度节点。前端看到的是 Python/NineToothed 计算过程，IR 应抽象其中的控制流、数据流、访存、规约、矩阵原语和数学函数，然后通过 pass 逐步加入 schedule、memory scope 和后端 intrinsic 信息，最后生成 Triton、CUDA、TileLang、TVM 等后端语言。

因此 canonical IR 的基本单元是：

- `arith.*`：标量/张量表达式。
- `math.*`：`exp/log/sqrt/rsqrt` 等数学函数。
- `reduce.*`：`sum/max/min` 等规约。
- `linalg.*`：`dot/matmul/transpose` 等细粒度线性代数原语。
- `scf.*`：循环、条件、yield 等控制流 region。
- `mem.*`：load/store、memory effect。
- `index.*`、`shape.*`：形状和索引计算。

`AttentionOpIR`、`FlashAttentionOpIR` 这类节点只能作为历史兼容层或 pattern 名称出现，不能成为 canonical SSA 的表达粒度。以 `tests/test_attention.py` 中的参考实现为例，canonical SSA 是 `scf.for + linalg.dot + reduce.max/reduce.sum + math.exp2 + select.where + mem.store`，不是一个 attention 节点。

## IR 分层

1. Frontend AST/Tensor Semantics

   输入是 NineToothed 的 arrangement/application 或已有结构化 ProgramIR。此阶段只负责理解用户代码与张量元信息。

2. Generic SSA

   语义层 IR，保持后端无关。它描述计算发生了什么，不描述 CUDA block、Triton program、TileLang fragment 或 TVM thread binding。

3. Canonical SSA

   通过 `ssa.canonicalize` 规范化 opcode、metadata 和 region 结构，为后续分析 pass 提供稳定输入。

4. Analyzed SSA

   `ssa.analyze_effects` 统计 store、reduction、loop、dot、online-softmax-shape 等事实。这里仍不做后端决策。

5. Scheduled SSA

   `ssa.select_schedule` 根据后端和分析事实加入 schedule annotation，例如 elementwise grid、parallel reduction、blocked linalg、online-softmax-dot。此阶段开始出现 tile size、vector width、threads、num_warps/num_stages 等策略，但仍保持为 IR metadata。

6. Target-Annotated SSA

   `ssa.lower_memory_scopes` 和 `ssa.lower_backend_intrinsics` 将抽象 schedule 映射到目标后端：

   - Triton：`tl.program_id`、`tl.load/tl.store`、`tl.dot`、Triton tensor/register 表达。
   - CUDA：`blockIdx/threadIdx`、thread-local register、`__shared__`、`__expf/expf`、后续 `mma.sync` 候选。
   - TileLang：`T.Kernel`、`T.get_thread_binding`、`local.fragment/shared/global`、`T.gemm/T.dot` 候选。
   - TVM：`T.thread_binding`、`T.alloc_buffer(scope="local/shared")`、tensorize intrinsic 候选。

7. Backend Source

   后端 emitter 读取 target-annotated SSA 和历史 ProgramIR 信息，生成目标语言源码。当前 CUDA/TileLang/TVM 的结构化算子 emitter 已能接收带 pass trace 的 SSA；Triton 普通算子仍可复用历史 Triton source path，但同样生成 SSA 审计信息。

## 可表达性

SSA 的表达能力来自三个方面：

- 数据流：每个 operation 显式列出 operands/results，支持 CSE、DCE、fusion、layout rewrite。
- 控制流：`scf.for/scf.if` 通过 region 表达循环和条件，loop-carried state 通过 block args 和 `scf.yield` 表示，能覆盖 online softmax、rowwise normalize、scan-like 更新。
- memory effect：`mem.store` 与 future `mem.load/alloc` 明确建模 side effect，便于在 schedule pass 中决定 coalescing、shared memory staging、local scratch placement。

这足以覆盖当前已验证的：

- elementwise/fill/copy。
- 1D reduction、axis reduction、rowwise softmax/layernorm 类融合。
- transpose、matmul。
- `tests/test_attention.py` 形式的 flash attention reference lowering。

## 可扩展性

新增后端时不需要改前端 IR，只需增加：

1. 目标 backend enum/registry。
2. backend-specific schedule policy。
3. memory scope mapping。
4. intrinsic mapping。
5. source emitter。

新增优化 pass 时应满足：

- 输入输出仍是 SSAProgramIR。
- pass 只处理细粒度 opcode，不按完整算子名改写。
- 若识别 pattern，也只能作为 schedule 选择依据，而不是替换成粗粒度语义节点。

## 当前实现状态

已新增 `ninetoothed.ssa_passes`：

```text
ssa.canonicalize
ssa.analyze_effects
ssa.select_schedule
ssa.lower_memory_scopes
ssa.lower_backend_intrinsics
```

公共 `ninetoothed.lowering.lower` 和直接 `ninetoothed.backends.lower(KernelIR, backend=...)` 都会运行该 pipeline。`BackendArtifact.metadata` 会带出：

- `ssa_pass_trace`
- `ssa_schedule`
- `ssa_metadata`
- `kernel_metadata`

## 后续优化方向

- Triton 普通算子 emitter 从历史 source passthrough 逐步迁移到 SSA source emission。
- CUDA matmul/attention 引入 shared memory tile、warp-level reduction、`mma.sync` tensorization pass。
- TileLang 引入 `T.gemm`/block fragment schedule，而不是 correctness-first scalar loops。
- TVM 引入 schedule rule/tensorize rule，生成标准 host wrapper 的环境兼容路径。
- 为 pass 增加 cost model/autotune metadata，并把测得性能反馈回 schedule selection。
