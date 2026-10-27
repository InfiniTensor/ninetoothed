# 项目结构与运行流程

本文按当前 `src/ninetoothed` 代码结构说明 NineToothed 的模块职责，并梳理从 Python kernel 定义到目标代码生成、运行时调用和 AOT 构建的主要流程。NineToothed 是一个面向张量计算的 Python DSL 与编译框架：用户用符号张量描述数据分块方式和每个计算单元的操作，编译器将 application 降为 SSA，再由目标后端生成并执行代码。

## 整体架构

一次典型编译会经过以下层次：

```text
用户 API (make / jit / build / aot)
    -> CompileRequest
    -> arrangement + application -> TensorSpec + SSA Program
    -> target 解析 -> SSA pass pipeline -> target validation
    -> Backend emitter -> Artifact (source + metadata)
    -> LaunchABI + LaunchPlan
    -> Materializer -> JIT callable 或 AOT binary
    -> runtime 参数绑定、校验、缓存和设备 launch
```

编译和物化是两个相邻但不同的阶段。`Compiler.compile` 负责生成 `Compilation`（IR、目标 Artifact、ABI 和 launch plan），不需要初始化目标运行时；`Compiler.materialize` 再选择对应 materializer，得到可调用的 kernel 或构建产物。`ninetoothed.compiler.lower` 可用于只生成 Artifact 的场景。

## 代码目录

### 顶层模块

`src/ninetoothed` 下的主要模块如下。少量工具模块（例如 `utils.py`、`naming.py`、`torchifier.py`）为这些主流程提供公共辅助功能。

| 路径 | 职责 |
| --- | --- |
| `__init__.py` | 包的公开入口，导出 `Tensor`、`Symbol`、`make`、`jit`、`build`、`eval` 等 API。 |
| `tensor.py`、`dtype.py`、`symbol.py` | 定义编译期符号张量、嵌套 dtype/形状表示、符号表达式和用于配置的 `block_size`。 |
| `language.py` | 提供 application 中可使用的张量运算、归约、点积等语言原语。 |
| `make.py`、`aot.py`、`build.py` | 传统 arrange-and-apply 构造、单 kernel AOT 接口、多配置构建和配置分发。 |
| `ascendifier.py`、`cudaifier.py` | 目标语言源代码适配工具；具体后端代码生成仍由 emitter 负责。 |
| `auto_tuner.py` | 对一组已构建候选 kernel 进行运行时测量和选择。 |
| `debugging.py`、`visualization.py` | 调试信息与 kernel/张量可视化支持。 |

### 前端与 IR

`frontend/` 和 `ir/` 定义了编译器的输入表示及语言边界：

- `frontend/python.py` 检查 application 的源代码和 AST，将 Python 子集转换为结构化 `ssa.Program`；不支持的语句或表达式会以 lowering 错误报告。
- `frontend/types.py`、`frontend/layout.py` 描述张量类型、布局和输入规格，并把符号 Tensor 视图转换为编译期 tensor specs。
- `frontend/errors.py` 统一前端错误类型；`frontend/__init__.py` 提供前端模块入口。
- `ir/ssa.py` 定义 SSA 值、类型、操作、基本块和程序等不可变 IR 结构。
- `ir/kernel.py` 定义 `Kernel`、`LaunchABI`、`LaunchBinding`、`LaunchPlan` 等编译和运行时共享的数据结构。
- `ir/frozen.py` 和 `ir/__init__.py` 提供 IR 的冻结/序列化等支持。

### 编译器

`compiler/` 是与具体设备后端相对独立的调度核心：

- `driver.py` 是统一编译入口。它组装 `CompileRequest`，解析目标，调用前端，运行后端 emit，并建立 ABI 与 launch plan。
- `passes.py` 实现 SSA pass 注册、pipeline 解析、通用分析/规范化、调度选择和目标能力校验。
- `specialization.py` 处理符号参数、张量规格和 schedule 参数的特化。
- `runtime.py` 把编译结果物化为 callable，负责运行时参数绑定、设备/dtype/布局约束检查、动态特化和 kernel launch。
- `cache.py` 提供基于内容摘要的 artifact/cache key、文件清单、原子写入和进程间锁。
- `layout.py`、`layout_runtime.py` 分别支持编译期布局/传输决策以及运行时布局约束验证。
- `jit.py` 是装饰器/函数式 JIT 接口；`reductions.py` 是通用归约相关编译支持。

### 后端

`backends/` 按代码生成、平台后端和运行时物化分层：

- `core.py` 定义 `Target`、`Artifact`、`Capability`、`Backend` 和 backend registry 的契约。
- `__init__.py` 创建默认 backend registry，并通过 `emit` 触发目标 lowering、校验和 emitter。
- `registry.py` 汇总各目标的 SSA pass 注册。后端的 schedule pass 注册到 compiler pass registry。
- `triton.py`、`cuda.py`、`tilelang.py`、`ascend.py` 定义目标 backend、能力、选项处理和目标 schedule 策略；Ascend/NPU 使用独立的 Ascend 后端，不替换 CUDA 后端。
- `emitters/` 将已经 lowering 的 Kernel/SSA 表示为目标源代码。通用分析、表达式上下文、SSA emitter 与 Triton/CUDA/TileLang/Ascend emitter 分别位于此处。
- `materializers/` 把 Artifact 变为可执行调用或可回载构建产物。每个 materializer 实现 JIT、AOT build、artifact reload 接口；默认分派注册在该目录的 `__init__.py`。
- `toolchain.py` 识别外部编译器及工具链信息；`triton_reductions.py` 为 Triton 目标提供特定归约调度支持。
- `templates/` 保存生成的 C++/运行时组件会使用的模板文件。

目标平台配置由 `targets.py` 解析为 `TargetContext`，其中包含 backend、platform、compute architecture 和平台能力元数据。选择 `backend` 决定语言/backend 路径；`platform` 与 `compute_arch` 用于具体目标约束。运行环境和目标平台不应混为一谈：编译器在 lowering 时解析目标快照，materializer/runtime 才与具体工具链和设备交互。

### 文档、测试与开发文件

- `docs/source/` 是 Sphinx 文档源文件，`docs/Makefile` 和 `docs/make.bat` 提供文档构建入口。
- `tests/` 覆盖前端 lowering、IR/pass、各后端 emitter、运行时、自动调优、AOT、缓存和算子行为；`tests/conftest.py` 与 `tests/utils.py` 提供测试 fixtures/工具。
- `scripts/` 放置贡献规范相关的检查脚本；`CONTRIBUTING.md` 描述贡献约定。
- `pyproject.toml` 定义包元数据、依赖和 Ruff 配置；`requirements.txt` 与 `docs/requirements.txt` 分别服务项目和文档依赖。

## 关键数据对象

`Tensor` 是编译期符号张量，不持有真实设备数据。`arrangement` 在符号张量上进行 `tile`、`expand`、`squeeze` 等元操作，形成嵌套张量视图；最外层 shape 用于计算 program 网格，内部层级由 application 的索引/循环访问。`application` 则描述每个 program 对应的块内计算。

前端会根据这些张量视图生成 tensor specs，并把 application 转为 `ssa.Program`。随后 `Kernel` 携带 SSA、张量规格、编译选项和元数据进入后端。最终 `Artifact` 保存目标源代码及其元数据；`LaunchABI` 描述公开参数如何映射成 kernel 参数，`LaunchPlan` 记录 grid、参数绑定、动态参数及可调候选等 launch 信息。

## 常见编译流程

### `make`：arrange-and-apply JIT

这是基础教程中的构造方式，适用于需要显式控制符号张量分块的 kernel：

```text
Tensor 参数模板
  -> arrangement(符号 Tensor)
  -> application AST lowering
  -> SSA pass pipeline
  -> Artifact / ABI / LaunchPlan
  -> backend materializer
  -> 首次调用时编译并执行，后续复用缓存
```

`ninetoothed.make(arrangement, application, tensors, ...)` 最终委托给 `compiler.driver.make`。默认 `caller="torch"` 选择 JIT 物化。目标通过 `backend` 指定，也可以由 `NINETOOTHED_BACKEND` 环境变量提供默认值。`num_warps`、`num_stages`、`pipeline`、`pass_options` 和目标专属选项会进入对应的编译阶段。

### `jit`：注解 Tensor 的直接编译

当 application 的参数已经用 Tensor 注解描述，可以使用 `@ninetoothed.jit`。此时 driver 从函数签名和注解读取张量规格，不经过独立 arrangement 函数：

```text
@jit(application)
  -> 读取签名与 Tensor 注解
  -> Python AST -> SSA
  -> 目标 lowering / emit
  -> materialize callable
  -> 运行时参数校验与 launch
```

`jit` 接受 `backend`、`platform`、`compute_arch` 和相同类型的 pass/后端配置。它和 `make` 共用 `CompileRequest`、编译 driver、后端 emitters 及 runtime。

### SSA lowering 与后端生成

编译 driver 的核心步骤可以拆为：

1. 根据 arrangement 生成符号张量视图，或从 jit 函数参数注解收集视图。
2. `tensor_specs` 从参数及视图生成类型、rank、shape、stride 等规格。
3. `from_application` 读取 application AST，生成细粒度 SSA 操作；这一步不做按算子名称的特殊路由。
4. 解析 `TargetContext`，归一化该目标的选项，并应用编译参数/运行时值特化。
5. backend `emit` 调用 `_prepare_kernel_for_backend`，为目标运行默认或用户指定的 pass pipeline，再执行目标能力验证。
6. 目标 backend/emitter 根据 lowering 后的 Kernel 生成 `Artifact` 和代码生成元数据。
7. driver 从张量规格、SSA 读写关系、目标 Artifact 和 meta 默认值推导 `LaunchABI` 及 `LaunchPlan`，并把 ABI/plan 写入 Artifact metadata。

默认 pass pipeline 由通用规范化、效果/调度分析、linalg decomposition、目标 schedule 选择/优化及能力验证等阶段组成。具体 pass 名称及选项可由 `pipeline` 和 `pass_options` 调整；后端通过注册表提供目标 schedule pass。目标若不支持程序使用的能力，应在目标校验阶段报错，而不是默默切换到另一目标。

### JIT 运行时与缓存

JIT materialization 由目标对应的 `Materializer` 实现。以 Triton 路径为例，流程为：

```text
Artifact source + Compilation metadata
  -> compilation_cache_key
  -> 写入共享 cache 的源文件 / manifest
  -> 导入目标模块并取得 kernel entrypoint
  -> 运行时绑定公开参数与 LaunchABI
  -> 校验 tensor/dtype/device/layout/别名约束
  -> 后端编译或命中设备编译缓存
  -> 根据 LaunchPlan 发起 kernel
```

编译缓存摘要覆盖源代码、SSA、launch plan、后端和目标元数据、编译/后端选项、pipeline、工具链身份及版本信息。默认缓存根目录是用户目录下的 `~/.ninetoothed`，可用 `NINETOOTHED_CACHE_DIR` 指定。对于需要运行时确定 dtype 或动态形状参数的路径，runtime 会按观测到的特化值创建并复用对应的编译实例。

### AOT 与多配置 build

`ninetoothed.aot.aot` 对单个已注解 application 走相同的 SSA 编译流程，但以 `mode="aot"` 调用 materializer，要求 `output_dir`，产出可部署的目标产物和清单。`load_built_artifact` 可由 `BuiltArtifact` 使用 backend materializer 回载已构建二进制；具体文件形式由目标 materializer 决定。

`ninetoothed.build` 面向 arrange-and-apply kernel 的多个预配置版本：

1. 对每个 `configs` 项调用 `premake`，得到 `(arrangement, application, tensors)`。
2. 每组通过默认 compiler 生成独立编译结果，并传入配置中的编译选项。
3. 非 meta 参数形成 variant dispatch key；同一个 key 下的 meta 候选组成 autotuner group。
4. 返回 dispatcher callable：按普通配置参数选择变体；有多个 meta 候选时，首次调用会 benchmark 并记录选择。
5. `lazy=True` 时，完整 build 推迟到 callable 第一次调用。

## 测试与定位问题

测试按功能大体分组，而不是仅按源目录一一对应：

- 前端与基础语义：`test_generation.py`、`test_lowering_inference.py`、`test_kernel_ir.py`、`test_ir_immutability.py`。
- SSA/pass 与目标调度：`test_ssa_pass_pipeline.py`、`test_ssa_validation.py`、`test_ssa_first_backend_lowering.py`、`test_*_reduction_schedule.py`。
- 后端、ABI、运行时和缓存：`test_backend_registry.py`、`test_target_profiles.py`、`test_compiler_entrypoints.py`、`test_compiler_cache_runtime.py`、`test_built_artifact_reload.py`。
- AOT/build/autotune：`test_aot.py`、`test_cpp_aot_support.py`、`test_triton_aot_materializer.py`、`test_auto_tuner.py`、`test_aot_auto_tuning.py`。
- 具体操作与算子：`test_add.py`、`test_matmul.py`、`test_softmax.py`、`test_attention.py` 等。
- Ascend/NPU：`test_ascend_backend.py`、`test_ascend_runtime.py`、`test_ascend_aot.py`。

推荐按故障发生阶段定位：AST/张量表示错误先看 `frontend` 和 lowering 测试；调度、能力或 pipeline 问题看 `compiler/passes.py` 与对应 backend pass 测试；生成源码问题看 `backends/emitters`；参数不匹配或执行错误看 `ir/kernel.py` 的 ABI/plan、`compiler/runtime.py` 和对应 materializer 测试。

本地基础检查示例：

```bash
python -m pytest tests/test_compiler_entrypoints.py
python -m pytest tests/test_ssa_pass_pipeline.py
python -m ruff check src/ninetoothed tests
```

需要 GPU/NPU 运行时的测试应在已安装匹配后端工具链且设备可用的环境中执行。纯 SSA、registry、IR 和许多 emitter 测试可用于不具备目标硬件的开发环境。

## 阅读代码的建议顺序

若要理解一条 kernel 的完整编译过程，建议从 `make.py` 或 `compiler/jit.py` 进入，接着读 `compiler/driver.py` 的 `_compile_kernel`，然后看 `frontend/python.py`、`compiler/passes.py`、目标 backend 和 emitter，最后读 `compiler/runtime.py` 与对应的 `materializers`。要理解多配置构建，可从 `build.py` 的 `build`、`_BuildHandle` 和 `_CandidateGroup` 开始。
