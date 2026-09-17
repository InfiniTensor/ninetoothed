# CPU 参考解释器与差分调试器：验收说明

本页是当前提交的验收入口。实现支持在不安装 Torch/Triton、CUDA 不可见的环境中执行 NumPy 参考语义，并提供逐阶段结果比较、SSA 差异定位、单步观察和失败回放。接口、支持矩阵和使用方法见[解释器文档](source/cpu_interpreter.rst)，独立安装见[安装说明](source/installation.rst#cpu-interpreter-without-gpu-packages)。

## 提交范围与原始证据

提交分支集中保留实现、测试、示例和文档。完整实验档案保存在个人仓库的固定提交中，不随本次功能改动复制到上游：

- [最终实现与实验说明](https://github.com/a962695448-rgb/ninetoothed/blob/0474ca792584a4b0b425abc1a271a6a974ac5740/docs/memory_checkpoints_20260914.md)
- [原始证据索引](https://github.com/a962695448-rgb/ninetoothed/blob/0474ca792584a4b0b425abc1a271a6a974ac5740/results/memory_checkpoints_20260914/README.md)
- [CPU 原始清单](https://github.com/a962695448-rgb/ninetoothed/blob/0474ca792584a4b0b425abc1a271a6a974ac5740/results/memory_checkpoints_20260914/cpu/ed33273/manifest.json)
- [A100 原始清单](https://github.com/a962695448-rgb/ninetoothed/blob/0474ca792584a4b0b425abc1a271a6a974ac5740/results/memory_checkpoints_20260914/gpu/ed33273/full/manifest.json)

上述归档对应旧计算源码 ed332733db28dbf16de06f166b16766760148958。本次提交已兼容上游 22e74c3 的多平台目标架构，当前 CPU 结果见[兼容性检查](upstream_compatibility_20260916.md)和[验证摘要](cpu_interpreter_validation.txt)。旧 A100 结果作为历史硬件证据保留，不代表合并后组合已经重新通过 GPU 验证。

## 功能与验证入口

| 能力 | 实现与测试 | 验收边界 |
|---|---|---|
| CPU 独立执行 | `interpret`、`interpret_program`；`test_interpreter_applications.py`、`test_interpreter_ssa.py` | 无 Torch/Triton 的环境执行 NumPy 语义，不回退 GPU；未支持语义明确失败 |
| 应用与 dtype | 逐元素、广播、非整除尾块、行归约、分支循环、softmax、dot/matmul | 必要 float32/int32/bool 语义；整数和布尔精确比较，float32 默认 `rtol=atol=1e-3`；更丰富 dtype 的范围见接口文档 |
| 默认优化管线逐阶段比较 | `check_passes`；`test_interpreter_default_pipeline.py`、`test_interpreter_provenance.py` | 同时比较相邻阶段与原始参考，检查累积漂移和相邻错误抵消；首次失败后停止后续 pass |
| SSA 差异定位 | `test_interpreter_value_mapping.py`、`test_interpreter_matmul_checkpoints.py`、`test_interpreter_failure_workflow.py` | 对齐 trace、保留操作及显式结果对应分别定位；实际矩阵乘分解检查左右取值、乘积及累加前缀 |
| 内存依赖 | `test_interpreter_memory_dependencies.py` | 活动字节区间连接同 program 最近写入；过滤记录、未知副作用、跨 program 顺序和重叠写 lane 保留明确边界 |
| 单步与观察 | `StepDebugger`；`test_interpreter_step_debugger.py`、[演示程序](cpu_interpreter_demo.py) | program ID/opcode 过滤，操作执行后的暂停、断点与 watch |
| 失败导出与回放 | `export_failure`、`replay_failure`；`test_interpreter_failure_workflow.py`、`test_interpreter_shared_storage.py` | 保存 SSA、数值输入、shape、dtype、seed、容差与诊断；保留共享视图；不导出任意 Python pass 代码 |
| 局部扩展 | 可注册 operation handler，共用 frontend/SSA 与内存模型 | 未声明的自定义副作用不能被当作完整内存历史；参见接口文档的扩展示例 |

默认管线包括 `ssa.canonicalize`、`ssa.analyze_effects`、`ssa.select_schedule`、后端的 `optimize_schedule`、`ssa.decompose_linalg` 和 `ssa.validate_target_capabilities`。CPU 执行这些 SSA 不能代替验证 GPU 代码生成和调度效果。

## 验证结果与统计口径

| 验证 | 已保存结果 | 口径 |
|---|---|---|
| 当前无 Torch/Triton 的 CPU 回归 | **577 passed，15 GPU deselected** | 合入 22e74c3 后的解释器/SSA/平台配置范围，含 9 项报告生命周期、10 项表达式计划、23 项 dtype 和新增 42 项恒等布局测试；不是无依赖的全仓库测试 |
| 本轮恒等布局内存及差分 | **三轮确认通过，15/15 GPU 差分、42/42 新增测试** | 受测大张量分配峰值下降 36%–42%；[范围与限制](cpu_interpreter_identity_memory.md) |
| 先前 dtype 性能及差分 | **三轮确认通过，15/15 GPU 差分、23/23 dtype 测试** | 多形状/步长/类型验证；[范围与数据](cpu_interpreter_dtype_performance.md) |
| 先前表达式计划性能及差分 | **三轮性能确认通过，15/15 GPU 差分、10/10 表达式测试** | 求值计划优化；[范围与数据](cpu_interpreter_performance.md) |
| 先前 RTX 4090 D 差分与报告协议 | **15/15 GPU 差分，9/9 报告测试** | 最终文件实机复验；九项测试使用替身验证协议；[详情](cpu_interpreter_report_validation.md) |
| 已归档 RTX 4090 测试集合 | **900 passed、2 skipped，覆盖 902 个测试 ID** | 1b68040；分段执行后逐项核对，含 15 项真实 GPU 差分；[详情](cpu_interpreter_rtx4090_validation.md) |
| 历史 A100-SXM4-40GB 完整回归 | **835 passed，2 skipped，570.53 秒，退出 0** | 旧计算源码 ed33273；包含 15 项实际 Triton GPU 差分；2 项跳过均要求至少双 GPU |
| 历史独立 CUDA dot | 四路正确性对照通过 | float32 标量分解案例，不代表完整 CUDA 后端或 Tensor Core 性能 |
| 已归档 wheel 与独立回放 | 68 份安装源码逐份核对，源码目录外的示例和独立回放通过 | `--no-deps` CPU 安装；保留原 GPU 依赖元数据，不是独立 CPU 发行包 |
| 文档与风格 | 归档中严格 Sphinx、Ruff、format、项目风格检查通过 | 当前整理后的检查另记在当前 CPU 验证摘要中 |

同一测试在不同阶段重复运行，不累计为更多独立用例。835 项里包含 77 项内存/检查点等新增测试。历史 ed33273 全库行覆盖率为 86.43%（10069/11650），不能当作解释器专项或分支覆盖率。之前的 1b68040 上游组合已经补充 RTX 4090 实机验证：同一源码和环境下，898 项通过与 2 项多 GPU 条件跳过，加上逐项补跑通过的 2 项，按测试 ID 核对后完整覆盖 902 项，不重复累计。中断、诊断进程退出和成功补跑的原始记录均保留。本次仍未新增 A100 运行，旧 A100 数据不自动覆盖当前组合。

### CPU 复现

在不含 Torch/Triton 的独立环境中，从仓库根目录执行统一入口：

~~~bash
python -m pip install -r requirements-cpu.txt
python scripts/run_cpu_tests.py --junitxml /tmp/nine-cpu-results.xml
~~~

该入口覆盖本页原有的解释器与 SSA 选择范围，并自动发现新增的解释器测试文件；Torch 适配专项单独排除，实际 GPU 差分仍明确取消选择。它会先检查 Torch/Triton 不存在，再设置 CPU 可见性和关闭第三方 pytest 插件，避免把有 GPU 包的环境误记为 CPU 独立验证。

新增 [CPU 自动回归工作流](../.github/workflows/cpu-interpreter.yml) 使用普通 Ubuntu 运行器，覆盖 Python 3.10/3.12 和 fork PR。工作流同时构建普通 wheel，在源码目录外执行已安装的示例与独立故障回放，保存 JUnit 和依赖版本。每次自动运行的结果以 Actions 记录为准，不把上表历史硬件结果计为新的 GPU 运行。

`test_interpreter_gpu.py` 也包含 CPU 可执行的 fixture/reference 检查，因此保留文件并明确取消选择真正的 GPU 测试。GPU 环境中的完整仓库命令是项目贡献指南规定的 `pytest`/doctest/coverage 流程；测试输出应在 PR 描述中明确附上，不把 fork PR 的跳过状态记成 GPU 通过。

## 实现取舍与已知限制

1. **有证据的定位优先。** `localization` 标明 `full_trace`、`aligned_prefix`、`retained_boundary` 或 `mapped_result`。操作来源只说明变换范围，不自动证明结果相等；没有可信对应关系的任意重构不能猜测唯一错误操作。最先观测到的 SSA 差异也不等于唯一缺陷根因。
2. **矩阵乘采用明确的标量契约。** 中间检查点来自原始输入的独立公式，要求有效 lane 按顺序完整执行 K。任意重排、split-K 或 FMA 不能直接沿用这一契约。步长变化破坏对应时，保留完整循环结果定位并报告原因。
3. **回放保留实际数值与存储关系。** 数值输入、共享视图、正负/零步长及只读权限经过验证后恢复；不使用 pickle 或执行导出的任意 Python 变换。变换本身抛出的异常保存为诊断，不能冒充可执行变换回放。
4. **CPU 顺序解释不模拟 GPU 竞争。** Warp/block 调度、shared memory、race condition 和多设备语义不在本解释器范围内；未支持的 atomics、间接指针、随机数、float8 等必须明确失败。
5. **按实际结果保留优化。** trace 索引在归档的 4096 映射 CPU 诊断基准中约快 33.84 倍；小规模末尾回溯存在约 0.037–0.142 ms 的退化。该数值只属于 CPU 诊断分析，不能写成 GPU 算子或端到端加速比。

本次没有自动缩减程序/shape，没有对所有输入、未来 pass 或唯一根因作普遍保证。故障注入用于测试定位能力，不冒充新发现的上游缺陷。

## 审查顺序与原有测试调整

建议先看 `interpreter/runtime.py` 与内存模型，再看 `ir/provenance.py`、`interpreter/debugger.py` 和定位/回放模块，最后对照专项测试。`frontend/preparation.py` 和相关调用方共享准备过程，避免 CPU 路径通过 GPU 后端执行。

当前差异保留了 CPU 可选收集、解释器相关回归，以及 jagged 的显式维度构造和独立参考检查。流上下文与确定性 scatter 的修正已由新上游提供，本次采用上游版本。未放宽计算容差，也未删掉断言获取通过；历史硬件记录与本轮 CPU 兼容性检查分别说明。

### GPU 报告保护

`python scripts/verify_interpreter_gpu.py --report /tmp/new-gpu-report.json` 要求输出路径不存在。脚本在导入 GPU 依赖之前独占创建文件，普通文件、目录和符号链接均不能被覆盖；重新验证请换用新路径。每个用例启动前写入进度，正常中断保存已完成结果并返回 130，状态为 `INTERRUPTED`；环境不可用为 `UNVERIFIED`（退出 2），数值失败为 `FAIL`（退出 1）。只有完整通过才写入 `PASS`（退出 0）。

`test_interpreter_gpu_report.py` 的 9 项 CPU 测试使用明确的测试替身检查报告协议，不构成 GPU 数值证据。进度文件会刷新到文件流；突然断电、SIGKILL 或写盘故障仍可能留下不完整文件，不能把 `RUNNING` 或损坏的文件认定为验收通过。
