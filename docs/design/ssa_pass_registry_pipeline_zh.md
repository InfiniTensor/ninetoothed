# 九齿 SSA Pass 注册器与优化遍执行流程

本文档描述当前九齿 SSA lowering 中的标准编译器式 pass 执行逻辑。目标是把优化遍从固定函数调用整理为可注册、可分类、可配置、可自动选择的 pipeline，为后续更多优化和自动调优留出接口。

## 设计目标

1. Pass 粒度保持在 SSA 计算、访存、控制和 schedule annotation 层，不引入 `AttentionOpIR` 之类粗粒度算子节点。
2. 优化遍按硬件无关和硬件相关分层执行。
3. 硬件相关优化可以按后端拆分，例如 `ssa.triton.optimize_schedule`、`ssa.cuda.optimize_schedule`。
4. 默认 pipeline 可以覆盖当前 Triton/CUDA/TileLang/TVM 后端。
5. 用户或 autotune 系统可以通过 declarative spec 替换 pass 序列和 pass 参数。

## Pass 分类

当前 pass registry 中的分类如下：

| 分类 | 作用 | 当前 pass |
| --- | --- | --- |
| `hardware_independent` | 不依赖后端，负责规范化和语义分析 | `ssa.canonicalize`、`ssa.analyze_effects` |
| `hardware_dependent` | 依赖后端能力，但多个后端共享同一 pass 框架 | `ssa.select_schedule`、`ssa.lower_memory_scopes`、`ssa.lower_backend_intrinsics` |
| `backend_specific` | 后端专属优化策略 | `ssa.triton.optimize_schedule`、`ssa.cuda.optimize_schedule`、`ssa.tilelang.optimize_schedule`、`ssa.tvm.optimize_schedule` |

默认 pipeline 形态为：

```text
ssa.canonicalize
-> ssa.analyze_effects
-> ssa.select_schedule
-> ssa.<backend>.optimize_schedule
-> ssa.lower_memory_scopes
-> ssa.lower_backend_intrinsics
```

## 执行流程

编译入口在 backend lowering 前调用 `lower_ssa_for_backend`。整体流程为：

1. 如果 `KernelIR` 只有 `ProgramIR`，先转换为 generic `SSAProgramIR`。
2. 根据目标后端、`compiler_options`、`kernel_metadata` 和显式参数选择 pipeline。
3. 通过 `SSAPassRegistry` 把 pass 名称解析为 descriptor 和 pass 实例。
4. 逐个执行 pass，并在 SSA metadata 中记录：
   - `pass_trace`
   - `pipeline_selection`
   - `schedule`
   - `optimization`
   - `memory_scope`
   - `backend_intrinsics`
5. 后端代码生成器读取这些 metadata 和 operation annotation，生成目标代码。

## 注册器接口

核心对象：

```python
from ninetoothed.ssa_passes import (
    DEFAULT_SSA_PASS_REGISTRY,
    registered_ssa_passes,
)
```

查询 pass：

```python
registered_ssa_passes(category="hardware_independent")
registered_ssa_passes(category="backend_specific", backend="triton")
```

新增 pass 的基本方式：

```python
from ninetoothed.ssa_passes import SSAPass, BACKEND_SPECIFIC
from ninetoothed.backends.base import BackendName


class MyTritonPass(SSAPass):
    name = "ssa.triton.my_pass"
    category = BACKEND_SPECIFIC
    phase = "optimization"
    supported_backends = (BackendName.TRITON,)

    def run(self, program, context):
        return program


DEFAULT_SSA_PASS_REGISTRY.register(MyTritonPass)
```

## 自定义 Pipeline

可以直接传入 `SSAPipelineSpec`：

```python
from ninetoothed.ssa_passes import SSAPipelineSpec, lower_ssa_for_backend

spec = SSAPipelineSpec(
    passes=(
        "ssa.canonicalize",
        "ssa.analyze_effects",
        "ssa.select_schedule",
        "ssa.triton.optimize_schedule",
        "ssa.lower_memory_scopes",
        "ssa.lower_backend_intrinsics",
    ),
    mode="custom",
    pass_options={
        "ssa.triton.optimize_schedule": {
            "tile": {"block_m": 64, "block_n": 32, "block_k": 32},
            "num_warps": 4,
        }
    },
)

target_ssa = lower_ssa_for_backend(generic_ssa, backend="triton", pass_pipeline=spec)
```

也可以通过 `KernelIR.compiler_options` 或 `KernelIR.metadata` 传入：

```python
compiler_options={
    "ssa_pass_pipeline": {
        "mode": "custom",
        "passes": (...),
        "pass_options": {...},
    }
}
```

## 自动调优入口

当前提供 policy-based autotune 入口：

```python
target_ssa = lower_ssa_for_backend(generic_ssa, backend="triton", autotune=True)
```

它会根据 SSA opcodes、program kind、目标后端选择候选 pipeline，并把选择结果记录到：

```python
target_ssa.metadata["pipeline_selection"]
```

当前 autotune 还不是运行时测量型，而是静态策略型。它已经具备三个关键扩展点：

1. `candidate_pipelines`：记录候选 pass 序列。
2. `pass_options`：记录候选配置，例如 tile、num_warps、num_stages。
3. `reason`：记录选择原因，便于报告和调试。

后续可以在这个接口上接入真实运行时 benchmark：

1. 生成多个 `SSAPipelineSpec`。
2. 对每个 spec 生成后端代码并运行 microbenchmark。
3. 缓存 shape/operator/backend 对应的最优 spec。
4. 以后遇到同类算子直接复用缓存 spec。

## 当前后端消费情况

Triton backend 已经实际消费 `optimization` metadata：

- elementwise/fill/copy：读取 `block_size`，使用更大的 coalesced vector block。
- matmul：读取 tile、num_warps、num_stages、input_precision、小问题阈值。
- 小矩阵使用 micro-kernel，中大矩阵使用二维 block + `tl.dot`。

CUDA/TileLang/TVM 目前也能收到后端专属 optimize pass 产生的 metadata，后续可以逐步把更多 schedule 决策接入各自代码生成器。

## 正确性约束

- pass registry 只控制 SSA pass 执行，不改变前端语义。
- backend-specific pass 只能添加或转换细粒度 SSA annotation，不允许把整个算子包成粗粒度节点。
- 所有 pass 执行结果都必须可序列化进 SSA metadata，便于审计、报告和自动调优缓存。
