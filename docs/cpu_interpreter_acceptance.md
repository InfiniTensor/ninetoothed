# CPU 参考解释器与差分调试器：验收说明

本页是当前提交的验收入口。实现支持在不安装 Torch/Triton、CUDA 不可见的环境中执行 NumPy 参考语义，并提供逐阶段结果比较、SSA 差异定位、单步观察和失败回放。接口、支持矩阵和使用方法见[解释器文档](source/cpu_interpreter.rst)，独立安装见[安装说明](source/installation.rst#cpu-interpreter-without-gpu-packages)。

## 提交范围与原始证据

提交分支集中保留实现、测试、示例和文档。完整实验档案保存在个人仓库的固定提交中，不随本次功能改动复制到上游：

- [最终实现与实验说明](https://github.com/a962695448-rgb/ninetoothed/blob/0474ca792584a4b0b425abc1a271a6a974ac5740/docs/memory_checkpoints_20260914.md)
- [原始证据索引](https://github.com/a962695448-rgb/ninetoothed/blob/0474ca792584a4b0b425abc1a271a6a974ac5740/results/memory_checkpoints_20260914/README.md)
- [CPU 原始清单](https://github.com/a962695448-rgb/ninetoothed/blob/0474ca792584a4b0b425abc1a271a6a974ac5740/results/memory_checkpoints_20260914/cpu/ed33273/manifest.json)
- [A100 原始清单](https://github.com/a962695448-rgb/ninetoothed/blob/0474ca792584a4b0b425abc1a271a6a974ac5740/results/memory_checkpoints_20260914/gpu/ed33273/full/manifest.json)

这些归档对应计算源码 `ed332733db28dbf16de06f166b16766760148958`。本次整理保持 `src/`、`tests/`、`scripts/` 和运行依赖内容与该版本一致；文档与仅服务于历史 `results/` 的检查配置另行整理。整理后的本地检查记录见[当前 CPU 验证摘要](cpu_interpreter_validation.txt)。

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

默认管线包括 `ssa.canonicalize`、`ssa.analyze_effects`、`ssa.select_schedule`、后端的 `optimize_schedule` 和 `ssa.decompose_linalg`。CPU 执行这些 SSA 不能代替验证 GPU 代码生成和调度效果。

## 验证结果与统计口径

| 验证 | 已保存结果 | 口径 |
|---|---|---|
| 无 Torch/Triton 的 CPU 回归 | **460 passed，15 GPU deselected** | ed33273 的选定 CPU 范围，不是无依赖的全仓库测试 |
| A100-SXM4-40GB 完整回归 | **835 passed，2 skipped，570.53 秒，退出 0** | 同一计算源码；包含 15 项实际 Triton GPU 差分；2 项跳过均要求至少双 GPU |
| 独立 CUDA dot | 四路正确性对照通过 | float32 标量分解案例，不代表完整 CUDA 后端或 Tensor Core 性能 |
| 普通 wheel 与独立回放 | 67 份安装源码逐份核对，25 个新故障包和 8 个历史故障包通过 | `--no-deps` CPU 安装；保留原 GPU 依赖元数据，不是独立 CPU 发行包 |
| 文档与风格 | 归档中严格 Sphinx、Ruff、format、项目风格检查通过 | 当前整理后的检查另记在当前 CPU 验证摘要中 |

同一测试在不同阶段重复运行，不累计为更多独立用例。835 项里包含 77 项内存/检查点等新增测试。全库行覆盖率为 86.43%（10069/11650），不能当作解释器专项或分支覆盖率。本次提交整理没有新增 A100 运行；上述 GPU 结果通过计算源码与测试内容的一致性建立对应关系。

### CPU 复现

从仓库根目录，使用安装了 NumPy、SymPy、pytest 的独立环境运行：

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' \
python -m pytest -q --color=no -ra --tb=short \
  tests/test_interpreter_applications.py \
  tests/test_interpreter_debugger.py \
  tests/test_interpreter_failure_workflow.py \
  tests/test_interpreter_gpu.py \
  tests/test_interpreter_ssa.py \
  tests/test_interpreter_step_debugger.py \
  tests/test_interpreter_default_pipeline.py \
  tests/test_interpreter_demo.py \
  tests/test_interpreter_matmul.py \
  tests/test_interpreter_provenance.py \
  tests/test_interpreter_value_mapping.py \
  tests/test_interpreter_memory_dependencies.py \
  tests/test_interpreter_matmul_checkpoints.py \
  tests/test_interpreter_shared_storage.py \
  tests/test_interpreter_trace_index.py \
  tests/test_ssa_application_lowering.py \
  tests/test_ssa_first_backend_lowering.py \
  tests/test_ssa_pass_pipeline.py \
  tests/test_ssa_program_domain_regressions.py \
  tests/test_ssa_validation.py \
  tests/test_ir_immutability.py \
  tests/test_kernel_ir.py \
  -k 'not test_cpu_interpreter_matches_actual_triton_gpu'
```

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

原有测试修改也保留在本次差异中：`tests/conftest.py` 允许缺少 Torch 的 CPU 选择；`test_aot.py` 使用正确的 `torch.cuda.stream(...)` 上下文管理器；`test_generation.py` 使用定义明确的输入/输出初值和不重复索引；`test_jagged.py` 使用可执行的参考构造。它们不改变 GPU 算子的计算容差，也不通过删掉失败断言获取通过。双卡测试的跳过条件与这些修正分别说明，不将二者混作同一改动。相关开发失败和复验原件仍在固定提交的历史档案内。
