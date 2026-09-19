---
name: "ntops-dev"
description: "Use this skill for any NineToothed/ntops operator task: implementing or integrating kernels, wrappers, exports, and tests; debugging correctness, broadcasting, dtype, non-contiguous stride/offset, generated source, AOT build, or performance regressions; and running fixed-configuration CUDA benchmarks. Trigger even when the user only names an operator, failing test, or benchmark problem."
---

# ntops 算子开发

目标是在 `ntops` 中完成可运行、可测试、改动范围清晰的算子补丁。默认只修改算子仓库；`ninetoothed` 用于查阅 DSL 和编译机制，除非任务明确要求，否则不改编译器核心。

## 默认范围

- 实现目录：`ntops/src/ntops/kernels/`、`ntops/src/ntops/torch/`、`ntops/tests/`。
- 参考目录：`ninetoothed/src/ninetoothed/`、`ninetoothed/docs/source/`。
- 不删除测试，不伪造 benchmark，不针对隐藏用例写死输入，不依赖私有服务。
- 算子开发任务必须由 NineToothed kernel 执行核心计算；不得用 wrapper 直接调用同名 `torch.*`、`Tensor.*` 或 PyTorch functional API 伪装成 ntops 实现。PyTorch 只作为测试参考实现，除非任务明确是 wrapper/fallback 集成而非 NineToothed 算子开发。
- 不为了让一个算子通过而修改 NineToothed 编译器核心；如果证据指向编译器限制，先记录最小复现、支持边界和规避方案。

## 任务分类与必查项

在修改文件前，将任务标记为下列一类或多类，并执行对应检查：

- 逐元素/广播：同 shape、标量/0 维 tensor、非规则 shape、广播方向、dtype promotion 和 mask。
- 归约/分块：归约维、非末维、单元素维、非 2 的幂长度、accumulation dtype 和数值稳定性。
- 布局敏感：`is_contiguous()`、stride、storage offset、transpose/slice 输入、输出布局契约和地址 mask。
- 性能/诊断/集成：正式 benchmark 协议、generated source、AOT build/导入、失败最小复现和候选保留/回退。

## 关键陷阱

- pytest 通过不能证明完成 NineToothed 实现；确认核心计算确实进入 kernel，而非 PyTorch 透传。
- 0 维 tensor 的 `.item()` 会触发 host sync。把它当作有明确性能限制的最后 correctness 规避，先验证 tensor/view 路径；`expand_as` 产生零 stride view，必须用真实 CUDA 用例验证。
- 不向零 stride 的 expanded **输出** view 写入多个逻辑结果；这会产生地址别名，且可能使 NineToothed 的 pointer/mask lowering 失败。输出 descriptor 与物理输出 rank/shape 对齐，并证明每个 program 只写一个明确的逻辑输出位置。
- 归约先比较两种架构：wrapper 将目标维移动到末维并规范化为固定 rank，或 arrangement 直接表达任意 rank/维度。只要契约允许，先实现 DSL 变换更少、可独立验证的固定-rank 方案；不要因参考算子更复杂就默认选择复杂映射。
- 布局测试必须比较输出；只断言 `not input.is_contiguous()` 不能证明算子正确。
- `git diff` 默认不包含未跟踪文件。封存补丁前检查 `git status --short`，并用 `git add -N <new-files>` 或工作树归档保留新文件。
- 最终摘要必须与最后一次命令日志一致；早期失败和后续成功都保留，但不把旧状态写成最终状态。

## 工作流程

1. 写清算子契约：输入、输出、shape、dtype、device、广播、非连续布局、stride/offset、边界条件、异常语义、不支持范围和性能目标。契约不清时先用 PyTorch 小实验确认，不边猜边实现。
2. 先找仓库中的相近实现：
   - 逐元素：`relu`、`silu`、`gelu`、`add`、`mul`。
   - 归约/分块：`softmax`、`max_pool2d`、`avg_pool2d`、`rms_norm`。
   - 布局敏感：`addmm`、`mm`、`bmm`、`conv2d`、`matmul`。
   - 性能/诊断：`scaled_dot_product_attention`、`matmul` 及其测试。
   - 打开相近算子的真实 kernel、wrapper 和测试；历史 example 只提供已观察到的证据，不能代替源码，也不能据其复杂度推断新算子的实现结构。
3. 只补任务需要的文件：
   - `src/ntops/kernels/<op>.py`
   - `src/ntops/torch/<op>.py`
   - 两个 `__init__.py` 中的导出
   - `tests/test_<op>.py`
   - 完成后用全新 Python 进程执行 `import ntops; getattr(ntops.torch, "<op>")`，防止只在当前进程注册成功而遗漏导出。
4. 优先复用仓库模式：
   - 普通逐元素算子使用 `ntops.kernels.element_wise.arrangement`。
   - wrapper 使用 `ntops.torch.utils._cached_make`。
   - 合适时使用 `tests.utils.generate_arguments()` 和 `skip_if_cuda_not_available`。
   - 归约先把“逻辑契约”与“kernel 物理布局”分开：在 wrapper 中规范化维度/shape，在 kernel 中只表达最小必要的 tile/reduce/store。只有 wrapper 规范化会破坏任务要求或带来不可接受复制时，才升级为任意维 stride-aware arrangement。
5. 先验证 correctness：
   - 参数化全量测试前，分别用 fresh process smoke：全归约、一个非末维归约、一个 non-contiguous/offset 输入。任何分支失败时先缩小到最小布局并与已通过分支比较，不在未理解地址映射时继续叠加 `permute`/`unsqueeze`/`expand`/`tile`。
   - 再跑 `python3 -m pytest tests/test_<op>.py -q`。
   - 修改公共辅助代码后，再跑相邻算子测试。
   - 参考结果使用 PyTorch API，并用 `torch.allclose`、`torch.testing.assert_close` 或精确比较。
   - 布局任务至少加入一个由 transpose 构造的 non-contiguous 用例。
   - 归约任务检查单元素归约维、非末维归约、大值和 dtype 敏感场景。
   - 数值失败先固定随机种子，不先放宽容差。
   - 先运行最小定向用例，再运行参数化用例和相邻回归；不用 `skip`/`xfail` 隐藏任务要求范围内的失败。
   - 除数值外检查 shape、dtype、device 和任务指定的 layout/异常语义。
6. Correctness 通过后再 benchmark：
   - 正式评测统一 `max_num_configs=1`，`premake()` 显式传入所评估的 `block_size`，不依赖自动调优隐式选配置。
   - 使用同一张 CUDA GPU、固定 shape、dtype、warmup 和重复迭代；基线与候选至少各运行 3 轮并比较 median。
   - 为减少温度和时序影响，最终复测使用基线/候选交错顺序；记录设备、环境、dtype、shape、命令、双方延迟和相对性能。
   - 改善不足 5% 或波动区间重叠时不宣称有效优化。
   - 没有 CUDA 时只记录待执行命令，不填写结果。
7. 性能明显落后时检查 generated source：
   - 运行 `scripts/inspect_generated_source.py <op>`。
   - 先在空或独立缓存中触发当前算子，或用 `--source` 指定本次生成文件；`--no-trigger` 返回的低/零匹配缓存不能作为当前算子的 generated-source 证据。
   - 记录源码路径、load/store、cast、autotune、tile/constexpr 等信号。
   - 给出一个有证据的瓶颈假设；没有复测结果时，不写“已优化”。
   - 每次只做一个可归因的最小候选修改。候选先通过 correctness，再按同协议复测；无收益、不稳定或破坏正确性时回退候选并记录拒绝原因。
8. AOT build/集成任务单独验收：
   - 记录完整 build 命令、环境变量、生成路径和导入/调用命令。
   - 区分 Python 注册、generated source 生成、AOT 编译、动态库加载和运行时错误，不用一个模糊的“build 失败”概括。
   - 若任务只要求修复配置或文档，保持最小补丁，不顺手改编译器核心。
9. 遇到失败时保存最小闭环：
   - 记录命令、报错、shape/dtype/device。
   - 说明根因判断和最小改动。
   - 重跑原失败用例和相邻回归。
10. 修改前检查常见 NineToothed 风险：
   - shape 派生的 block size 使用 `Symbol(..., constexpr=True)` 或调用参数，不按单一 shape 写死。
   - `broadcast`、`expand`、`.offsets()`、`.data_ptr()`、scatter/store 等任务先核对 arrangement、地址和 mask。
   - `eval(subs)` 失败时，将其作为诊断线索，并补一个小型 CUDA reference。
   - `torch.compile` 任务应稳定 custom-op schema，并尽量在图外预编译或缓存 kernel。

## 按需资料

- 文件布局和补丁格式：`references/ntops_operator_workflow.md`
- DSL 与 arrangement：`references/ninetoothed_dsl_quick_reference.md`
- Correctness 和 benchmark 记录：`references/validation_and_benchmark.md`
- Generated source 排查：`references/generated_source_diagnosis.md`
- 已知九齿问题模式：`references/feishu_ninetoothed_fix_notes.md`

## 可复用脚本

- 聚焦 correctness：`scripts/run_correctness_test.py --help`
- 通用 3 轮交错计时与 promotion 判定：在任务 benchmark 中导入 `scripts/benchmark_protocol.py`
- 已支持算子的快速 benchmark：`scripts/run_operator_benchmark.py --help`
- generated source 生成/指定文件/最新缓存检查：`scripts/inspect_generated_source.py --help`
- 布局、归约和数值复现：`scripts/run_coverage_checks.py --help`

## 历史案例路由

不要默认扫描所有 examples。先读当前仓库真实源码；仅在需要同类失败证据时再读取一个最相关案例。案例是历史证据，不是实现模板或待复制答案：

- 0 维/标量失败：`examples/maximum_new_operator/`
- 归约与 generated source：`examples/softmax_reduce_selftest/`
- non-contiguous/布局：`examples/addmm_layout_selftest/`
- 数值与环境失败：`examples/matmul_sdpa_diagnosis/`
- 多算子性能差距：`examples/official_ntops_benchmark/`

## 完成条件

- 算子契约和不支持范围已写清。
- 已检查相近实现。
- kernel、wrapper、导出和测试齐全。
- 核心计算由 NineToothed kernel 完成，没有同名 PyTorch API 透传。
- 已记录 correctness 命令和真实结果。
- 性能敏感任务已在 `max_num_configs=1` 下记录正式 benchmark；无法执行时明确标为待验证。
- 性能回退有 generated-source 分析或可检验的原因判断。
- 候选修改有接受/回退证据，无收益候选已回退。
- dtype、边界、布局或固定种子用例与任务风险相匹配。
- 补丁没有无关重构，失败和性能差距没有被隐藏。

最终交付按以下顺序摘要，并确保每个结果都能在原始日志中找到：

```text
Files changed:
Correctness command/result:
Adjacent regression command/result:
Benchmark protocol/result:
Generated-source finding:
Candidate accepted/reverted and why:
Known limitations or unresolved failures:
```
