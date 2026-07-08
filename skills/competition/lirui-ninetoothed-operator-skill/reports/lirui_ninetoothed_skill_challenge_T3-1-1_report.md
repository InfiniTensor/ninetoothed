# Lirui NineToothed Skill Challenge T3-1-1 Report

## 1. 摘要

本作品是一个面向 NineToothed 算子开发的 `.skill` 工作流包，目标是指导 AI 编程智能体完成需求理解、实现规划、正确性测试、benchmark、generated source / AOT build 检查、失败诊断与 PR 集成。作品放置在 `skills/competition/lirui-ninetoothed-operator-skill/`，符合操作手册中对比赛作品目录的要求。

本作品不是隐藏任务答案集，不包含硬编码隐藏评测任务，不包含 API key 或账号凭据，不绕过测试，也不伪造 pytest 或 benchmark 结果。所有未真实执行的结果均保留为 `待真实运行后填写`。

## 2. 赛题理解

九齿 `.skill` 创新挑战赛道的核心不是提交某个固定算子答案，而是提交一个可复用的 AI 编程工作流。这个 workflow 应帮助 Trae、Cursor、Codex 等 AI 编程工具更稳定地完成 NineToothed 仓库里的算子开发、测试、调试、benchmark 和 PR 集成任务。

根据操作手册，最终作品应通过 NineToothed 主仓库 PR 提交，并放在：

```text
skills/competition/<skill-name>/
```

本作品使用的目录为：

```text
skills/competition/lirui-ninetoothed-operator-skill/
```

## 3. .skill 目标、适用范围与不适用范围

目标：

- 让 AI 智能体先阅读仓库上下文，再开始实现。
- 把 NineToothed 算子任务拆成 requirement extraction、arrangement、application、Tensor specs、correctness、benchmark、AOT、diagnosis 和 PR integration。
- 强制记录真实命令和真实结果。
- 强制 layout-sensitive 测试覆盖非连续输入。
- 强制 benchmark 结论包含 baseline、input size、dtype、layout、command 和 numeric result。

适用范围：

- NineToothed operator development.
- Correctness test.
- Benchmark.
- Generated source inspection.
- AOT build.
- Failing test diagnosis.
- PR integration.

不适用范围：

- 隐藏评测答案。
- 硬编码隐藏任务名。
- 无关仓库重构。
- 私有依赖、API key 或账号凭据。
- 删除测试、绕过测试或伪造结果。

## 4. 包结构与核心文件说明

```text
skills/competition/lirui-ninetoothed-operator-skill/
├── SKILL.md
├── README.md
├── references/index.md
├── scripts/run_self_tests.sh
├── scripts/collect_logs.py
├── examples/task1_elementwise_broadcast.md
├── examples/task2_reduction_block.md
├── examples/task3_layout_sensitive.md
├── examples/task4_benchmark_debug.md
├── tests/validation_plan.md
├── reports/lirui_ninetoothed_skill_challenge_T3-1-1_report.md
├── HONOR_CODE.md
└── REFERENCE.md
```

核心文件：

- `SKILL.md`: 可执行 SOP，定义何时使用、第一步读什么、如何抽取算子需求、如何测试、如何 benchmark、如何诊断失败、如何准备 PR。
- `README.md`: 面向评委的总览，包含 self-test task matrix。
- `references/index.md`: 基于仓库文件整理的 repo map、关键 DSL 文件、测试样例、AOT/build/generated source 和 CONTRIBUTING checklist。
- `examples/`: 四个具体自测计划。
- `tests/validation_plan.md`: 验证 skill 有效性的计划。
- `HONOR_CODE.md` 与 `REFERENCE.md`: 诚信、引用和 AI 辅助披露。

## 5. 核心工作流

本 skill 要求 AI 智能体按以下顺序行动：

1. 阅读 `README.md`、`CONTRIBUTING.md`、`docs/`、`tests/` 和相关 `src/ninetoothed/` 实现。
2. 搜索相似 `arrangement`、`application`、`Tensor`、`ninetoothed.make`、`ninetoothed.build` 模式。
3. 抽取 operator requirements，包括 inputs、outputs、shape、dtype、broadcast rule、mask、boundary cases、tolerance 和 layout constraints。
4. 设计 layout checklist，至少包含一个 non-contiguous input 测试。
5. 选择最小可测试实现路径。
6. 使用 PyTorch reference 或仓库已有 reference 做 correctness test。
7. 记录 pytest 命令和真实结果。
8. 设计 benchmark，记录 baseline、input size、dtype、layout、command、result 和 conclusion。
9. 对失败记录 symptom、error message、suspected root cause、minimal fix、re-run command 和 re-run result。
10. 只改允许目录，按 CONTRIBUTING 运行 Ruff、style checker 和 pytest，并在 PR 描述中包含 pytest output。

## 6. 为什么选择这四类自测任务

四个 self-test tasks 对应 NineToothed 算子开发中最容易出错、也最可能出现在隐藏评测中的能力维度：

- T1 elementwise / broadcast: 覆盖最基础的 block tiling、broadcast alignment、partial block mask 和 PyTorch reference 比较。
- T2 reduction / block: 覆盖 `ntl.sum`、`ntl.max`、稳定 softmax、accumulator dtype、neutral `other` value 和非整除 reduction size。
- T3 layout-sensitive: 覆盖 transpose、slice、stride、storage offset、view 和 non-contiguous input，防止只在 contiguous tensor 上通过。
- T4 benchmark / debug / AOT: 覆盖 AOT build、generated source、auto-tuning CSV、cache fingerprint、真实 benchmark 记录和失败诊断。

这四类任务不针对隐藏任务名称，而是覆盖隐藏评测可能考察的通用能力：正确性、泛化性、layout 鲁棒性、性能证据、AOT 集成和失败修复过程。

## 7. 自测任务 1：elementwise / broadcast

文档位置：`examples/task1_elementwise_broadcast.md`

建议任务：broadcast add 或 masked add。

覆盖点：

- `tests/test_add.py` 的 elementwise add 模式。
- `tests/test_addmm.py` 的 scalar 参数和 PyTorch reference 模式。
- `tests/test_pow.py` 的 elementwise scalar-like 参数模式。
- shape broadcast、singleton dimension、non-power-of-two size 和 partial block。

隐藏评测相关性：很多未知算子可以退化为 elementwise + broadcast + mask。该任务验证 AI 是否会先明确 broadcast rule 和 mask，而不是写只适用于同 shape contiguous 输入的实现。

真实本地检查结果：

```powershell
$env:PYTHONPATH = "$PWD\src"
pytest tests/test_add.py tests/test_addmm.py tests/test_pow.py -q
```

结果：pytest collection failed because the local Windows Python environment could import `ninetoothed` from `src`, but `ninetoothed` imports Triton and Triton is not installed.

关键错误：

```text
ModuleNotFoundError: No module named 'triton'
```

涉及文件：

- `tests/test_add.py`
- `tests/test_addmm.py`
- `tests/test_pow.py`
- `src/ninetoothed/generation.py`

解释：这是本地环境限制导致的 collection failure，不是 elementwise / broadcast correctness failed。

日志路径：

```text
skills/competition/lirui-ninetoothed-operator-skill/reports/logs/20260708-103141_pytest_t1_missing_triton_pytest_t1_missing_triton.log
```

TODO: 在 Linux/WSL/CUDA/Triton 环境安装仓库依赖后重新运行同一 focused pytest 命令。
## 8. 自测任务 2：reduction / block

文档位置：`examples/task2_reduction_block.md`

建议任务：reduce sum、block max 或 softmax-like reduction。

覆盖点：

- `tests/test_softmax.py` 中 `Tensor(2, other=float("-inf")).tile((1, BLOCK_SIZE))`、`ntl.max`、`ntl.exp`、`ntl.sum` 的 reduction 模式。
- `tests/test_attention.py` 中 block computation 和 reduction-like accumulation。
- 非 power-of-two reduction dimension，例如 `(781, 1823)`。
- accumulator dtype、数值稳定性和 tolerance。

隐藏评测相关性：隐藏任务可能包含 reduce、normalize、softmax、argmax-like 或 block aggregation。该任务验证 AI 是否能处理 reduction axis、partial block neutral value 和精度容差。

当前真实运行结果：

```text
待真实运行后填写
```

## 9. 自测任务 3：layout-sensitive

文档位置：`examples/task3_layout_sensitive.md`

覆盖点：

- transpose: `torch.randn((n, m), device=device, dtype=dtype).t()`。
- slice: `torch.randn((m, n * 2), device=device, dtype=dtype)[:, ::2]`。
- storage offset: `torch.randn((m + 2, n + 2), device=device, dtype=dtype)[1:-1, 1:-1]`。
- non-contiguous check: `assert not tensor.is_contiguous()` where applicable。
- stride and offset logging: `tensor.stride()` and `tensor.storage_offset()`。

参考仓库文件：

- `tests/test_matmul.py`。
- `tests/test_conv2d.py`。
- `tests/test_clone.py`。
- `tests/test_getitem.py`。
- `src/ninetoothed/tensor.py`。
- `src/ninetoothed/generation.py`。

隐藏评测相关性：隐藏测试很可能包含非连续输入、转置输入、切片输入或 view 输入。如果 skill 不强制 layout 测试，AI 可能只在 contiguous happy path 上通过。

当前真实运行结果：

```text
待真实运行后填写
```

## 10. 自测任务 4：benchmark / debug / AOT build

文档位置：`examples/task4_benchmark_debug.md`

覆盖点：

- `docs/source/build.rst`: `premake`、`configs`、`meta_parameters`、`output_dir`、lazy build、generated artifacts 和 caching。
- `tests/test_aot.py`: `caller`、`kernel_name`、`output_dir` 的 AOT 调用。
- `tests/test_aot_auto_tuning.py`: `ninetoothed.build`、auto-tuning 和 CSV。
- `src/ninetoothed/aot.py`: AOT dispatcher、`.so` compilation 和 launch wrapper。
- `src/ninetoothed/build.py`: multi-config build、auto-tuning、fingerprint、cache fallback。
- `src/ninetoothed/generation.py`: generated source cache。

隐藏评测相关性：隐藏任务可能不只看正确性，还看是否能定位 AOT build、generated source、cache 或 benchmark 问题。该任务验证 AI 是否能提供可复现证据，而不是写“性能很好”。

当前真实运行结果：

```text
待真实运行后填写
```

## 11. benchmark 设计

本作品至少设计两个 benchmark，但不填写伪造数字。

### Benchmark A: Elementwise broadcast runtime

- Baseline: PyTorch `torch.add`。
- Candidate: NineToothed broadcast add or masked add。
- Input sizes: `(1024, 1024)` and `(4096, 257)`。
- Dtype: `float32`; optional `float16` with tolerance。
- Layout: contiguous plus optional non-contiguous sliced case。
- Required record: command, timing unit, result, conclusion。

Result:

```text
待真实运行后填写
```

### Benchmark B: AOT vector add and auto-tuning

- Baseline: PyTorch `torch.add` or a fixed-block NineToothed AOT variant。
- Candidate: `ninetoothed.build` with multiple `block_size` configs and `meta_parameters=("block_size",)`。
- Input sizes: `(1127,)` and `(20260128,)` following the style of `tests/test_aot_auto_tuning.py`。
- Dtype: `float32`; optional `float16`。
- Layout: contiguous。
- Required generated-source evidence: `.cpp`, `.h`, `.so`, `.csv`, `.fingerprint` presence in intended `output_dir`。

Result:

```text
待真实运行后填写
```

## 12. 失败诊断案例或模板

失败诊断必须记录：

- Symptom.
- Error message.
- Suspected root cause.
- Minimal fix.
- Re-run command.
- Re-run result.

Planned examples:

- T1: wrong broadcast `expand` causing shape mismatch.
- T2: wrong neutral `other` value causing partial-block reduction mismatch.
- T3: non-contiguous input fails because stride or offset was implicitly assumed contiguous.
- T4: `output_dir` missing or stale cache causes AOT build/load failure.

Current real case:

- Symptom: pytest collection failed.
- Error message: `ModuleNotFoundError: No module named 'triton'`.
- Root cause: local environment lacks Triton dependency required by the `ninetoothed` import path.
- Minimal fix: install a supported Triton environment or rerun inside Linux/WSL/CUDA environment with repository dependencies installed.
- Re-run command: `pytest tests/test_add.py tests/test_addmm.py tests/test_pow.py -q`.
- Re-run result: 待真实运行后填写.

## 13. skill vs no-skill 对比

计划对比同一任务在未使用 skill 和使用 skill 两种情况下的表现：

- 是否先抽取 requirements。
- 是否搜索相似仓库实现。
- 是否覆盖 broadcast、partial block、non-contiguous layout。
- 是否使用 PyTorch reference。
- 是否记录 pytest 命令和真实结果。
- 是否设计 benchmark 并记录 baseline、input size、dtype、layout、command 和 numeric result。
- 是否遵守只改 `skills/competition/lirui-ninetoothed-operator-skill/` 的范围要求。

Result:

```text
待真实运行后填写
```

## 14. 安全、依赖、授权、引用与 AI 辅助披露

本作品不包含 API key、账号凭据、隐藏评测答案、硬编码隐藏任务名、绕过测试逻辑或伪造结果。引用资料和 AI 辅助范围见 `REFERENCE.md`。诚信声明见 `HONOR_CODE.md`。

如果后续执行测试或 benchmark，所有输出应来自真实命令，并可通过 `scripts/collect_logs.py` 归档到 `reports/logs/`。

## 15. 本地检查记录

本轮记录真实本地检查结果，不伪造 pytest 通过，也不填写未运行的 benchmark 数字。

| Check | Real local result | Interpretation |
| --- | --- | --- |
| `python skills/competition/lirui-ninetoothed-operator-skill/scripts/collect_logs.py --help` | Passed; help output includes `Copy a test or benchmark log into the skill reports/logs directory.` | `collect_logs.py` 可用。 |
| `ruff check` | `All checks passed!` | 静态检查通过。 |
| `python scripts/check_contributing_style.py` | No output | 按仓库脚本习惯记录为无错误输出。 |
| `ruff format --check` | Initially reported `collect_logs.py` would be reformatted | Format issue found and fixed by `ruff format skills/competition/lirui-ninetoothed-operator-skill`. |
| T1 focused pytest | Collection failed with `ModuleNotFoundError: No module named 'triton'` | 本地 Windows Python 环境缺少 Triton；这是环境限制，不是 correctness failure。 |

T1 failure diagnosis:

- Symptom: pytest collection failed.
- Error message: `ModuleNotFoundError: No module named 'triton'`.
- Root cause: local environment lacks Triton dependency required by the `ninetoothed` import path.
- Minimal fix: install a supported Triton environment or rerun inside Linux/WSL/CUDA environment with repository dependencies installed.
- Re-run command: `pytest tests/test_add.py tests/test_addmm.py tests/test_pow.py -q`.
- Re-run result: 待真实运行后填写.

Benchmark status:

```text
待真实运行后填写
```

## 16. 局限性与后续维护计划

局限性：

- 已记录 T1 focused pytest 的真实 collection failure；T2/T3/T4 和完整 pytest 仍需在合适环境中补充。
- benchmark 数字需要在真实 CUDA/Triton/NineToothed 环境中运行后补充。
- AOT build 检查依赖 CUDA、Triton、NVCC 和本地 GPU 环境。
- `skill vs no-skill` 对比需要设计同一任务的双运行实验。

后续维护计划：

- 用真实 self-test 执行结果补全四个 examples。
- 收集 pytest、Ruff、style checker 和 benchmark 日志。
- 根据真实失败案例补充 diagnosis patterns。
- 随 NineToothed `docs/`、`tests/` 和 `CONTRIBUTING.md` 更新维护 references。


