# 九齿 .skill 创新挑战 Proposal

## 1. 基本信息

- 选手姓名：`赖泉laiquan`
- 报名赛题编号：T3-1-1
- 报名赛题名称：NineToothed 算子开发 Skill
- 拟提交 `.skill` 名称：`ninetooth-operator-dev`
- 拟提交目录：`skills/competition/ninetooth-operator-dev/`
- Proposal 日期：2026-05-19

## Final submission status（最终提交状态，2026-07-14）

本 proposal 记录的是初始设计和量化目标。第 9 节中的通过率、benchmark
数量和隐藏任务分数是计划目标，not achieved results，不能作为最终实测结果
引用。

最终提交已完成 skill 包、结构校验、提交包回归测试和静态检查。当前 macOS
环境为 Python 3.11.6、PyTorch 2.10.0，`torch.cuda.is_available()` 为
`False`。执行 `python -m pip install "triton>=3.0.0"` 时没有可用的 macOS
发行包；执行上游全量 `pytest` 时因 `ModuleNotFoundError: triton` 产生
23 collection errors。因此四个算子自测和 benchmark 保留 `Not run yet`，
最终材料只提交可审查的任务规格和明确的运行时 blocker，不声称已完成 GPU
correctness 或性能验证。

## 2. .skill 目标、目标用户与目标任务

`ninetooth-operator-dev` 的目标是把 NineToothed 算子开发过程中容易散落在代码、测试、benchmark、generated source 和调试经验里的隐性知识，沉淀成一个可安装、可离线复现、可被 AI 智能体稳定调用的 `.skill`。

目标用户是需要在 NineToothed 仓库中完成算子开发、测试、性能分析和失败诊断任务的 AI 智能体。该 `.skill` 不直接替代 AI 智能体写代码，而是提供任务拆解流程、仓库阅读路径、算子实现模式、测试与 benchmark 模板、失败诊断清单和报告格式，使智能体能在统一时间预算下更稳定地完成隐藏评测任务。

目标任务包括：

- 根据算子需求提取输入、输出、shape、dtype、broadcast、mask、边界条件和布局约束。
- 阅读并复用 NineToothed 现有 arrangement、application、tensor meta-operation、load/store、generated source、测试和示例代码。
- 选择符合仓库风格的 NineToothed DSL 表达方式，完成最小范围实现。
- 编写 correctness test，并与 PyTorch 或仓库已有参考实现对齐。
- 针对性能敏感任务补充 benchmark、检查 generated source 或 AOT build 输出，并判断是否存在明显回退。
- 在失败时记录现象、诊断路径、根因、最小修复或规避建议，以及验证命令和结果。
- 明确说明不支持场景，避免对 dtype、动态 shape、非连续布局或硬件能力做过度承诺。

## 3. 设计原则

1. 渐进披露：`SKILL.md` 只保留触发条件、核心工作流、约束和验证闭环；较长的 DSL 模式、测试矩阵、benchmark 说明和失败案例放入 `references/`，由智能体按需读取。
2. 评测对齐：工作流直接映射隐藏任务评分项，包括任务完成度、测试验证、性能意识、补丁最小性、仓库风格一致性、过程与合规。
3. 仓库优先：所有实现选择优先模仿 NineToothed 仓库已有代码、命名、构建、测试和 benchmark 方式，避免引入新的抽象体系。
4. 最小补丁：智能体必须优先做局部修改，不做大面积格式化、不改编译器核心、不删除测试、不绕过失败。
5. 离线可复现：`.skill`、示例、脚本和自测材料不依赖在线服务、私有账号或 API key；外部依赖必须说明版本、用途和 fallback。
6. 失败可诊断：每个任务都要求留下可复查执行摘要，包括读过的关键文件、运行过的命令、失败现象、修复理由和验证结果。

## 4. 计划覆盖范围

### 4.1 算子类型

| 类别 | 计划覆盖内容 | 典型任务 |
| --- | --- | --- |
| 逐元素 / 广播算子 | dtype 约束、broadcast、mask、边界处理、标量参数 | add、relu、gelu、bias + activation |
| 归约 / 分块算子 | reduce 维度、tile/block 配置、局部统计、数值稳定性 | softmax、row max、sum/reduce、max_pool2d 子任务 |
| 布局敏感算子 | non-contiguous、stride、offset、contiguous fallback、边界索引 | strided unary、strided add、offset slice 场景 |
| 性能 / 诊断 / 集成任务 | generated source 检查、AOT build、benchmark 对比、性能回退定位 | 冗余 load/store 分析、AOT 配置修复、benchmark 补齐 |
| 文档 / 示例补齐 | application 示例、README 片段、任务报告摘要 | 为新增算子补示例和使用说明 |

### 4.2 输入布局与边界条件

计划覆盖以下输入场景：

- contiguous tensor 与常规 dense layout。
- non-contiguous view、显式 stride、offset 或 sliced input。
- 广播维度，包括标量广播、尾维广播和 batch 维广播。
- 常见 dtype，包括仓库已有支持的 float32、float16、int 类型或 bool/mask 类型；最终支持范围以 NineToothed 仓库实际能力为准。
- 边界 shape，包括空维度或最小尺寸、非 2 的幂尺寸、不能整除 tile/block 的尾块、极端 batch 或 channel 尺寸。

不计划覆盖需要修改 NineToothed 编译器核心机制的任务，也不把未知硬件后端或仓库未支持 dtype 作为承诺范围。

### 4.3 性能验证方式

`.skill` 会引导智能体使用三层性能验证：

1. 生成代码检查：查看 generated source 是否存在明显冗余 load/store、重复广播计算、无效边界判断、未使用 stride/contiguous 信息等问题。
2. benchmark 对比：在至少两组输入规模上比较新增实现与仓库已有实现、PyTorch 参考或上一版实现的耗时差异，并记录运行命令、硬件信息、输入规模和统计方式。
3. 回退诊断：若性能明显落后，优先定位为布局处理、tile/block 选择、AOT build 配置、内存访问模式、dtype 转换或边界分支问题，并给出最小修复或明确规避建议。

### 4.4 调试场景

计划覆盖的失败诊断场景包括：

- correctness test 失败：shape/dtype 推断错误、broadcast 维度错误、边界 mask 错误、数值容忍度设置不合理。
- non-contiguous 失败：stride 或 offset 未被正确用于 load/store，contiguous 假设泄漏。
- generated source 异常：生成代码缺失、符号命名不一致、冗余访存、边界判断过多。
- AOT build 失败：构建目标、路径、编译参数或依赖配置错误。
- benchmark 结果异常：输入规模不稳定、warmup 不充分、基线不公平、统计口径不一致。
- 集成失败：测试文件位置、示例入口、命名风格或导出路径不符合仓库惯例。

## 5. 交付包结构

```text
ninetooth-operator-dev/
  SKILL.md
  README.md
  README_zh.md
  HONOR_CODE.md
  REFERENCE.md
  FINAL_REPORT.md
  proposal.md
  agents/
    openai.yaml
  references/
    repo-map.md
    operator-task-contract.md
    dsl-pattern-index.md
    verification-matrix.md
    performance-diagnostics.md
    failure-playbook.md
    subagent-orchestration.md
    entropy-gc.md
    script-index.md
  scripts/
    collect_repo_map.py
    lint_skill_structure.py
    scaffold_selftest.py
    scaffold_subagent_session.py
  examples/
    elementwise-broadcast/TASK.md
    reduction-block/TASK.md
    layout-sensitive/TASK.md
    performance-diagnostics/TASK.md
  subagent-sessions/
    _template/
  tests/
    eval_cases.yaml
    test_structure.py
    test_submission_package.py
```

核心文件说明：

- `SKILL.md`：声明触发场景、端到端工作流、必须检查项、禁止行为和最终输出格式。
- `references/repo-map.md`：说明智能体进入 NineToothed 仓库后应优先搜索和阅读的目录、文件模式与关键术语。
- `references/operator-task-contract.md`：把自然语言算子需求转成 shape、dtype、layout、边界条件和验收标准。
- `references/dsl-pattern-index.md`：总结逐元素、broadcast、reduce、tile/block、load/store、stride/offset 等 DSL 表达模式。
- `references/verification-matrix.md`：提供 correctness 测试矩阵、PyTorch 对齐策略和交付证据要求。
- `references/failure-playbook.md`：提供测试失败、generated source、AOT build、benchmark 回退的诊断路径。
- `scripts/`：提供离线结构校验、仓库地图收集、自测脚手架和子智能体会话脚手架。
- `examples/`：包含 4 个完整自测任务规格，覆盖逐元素/广播、归约/分块、布局敏感和性能诊断。
- `tests/eval_cases.yaml`：定义自测任务元数据和评分 rubrics，便于复跑和对比。
- `tests/test_structure.py` 与 `tests/test_submission_package.py`：验证 skill 结构、提交材料、身份字段、链接和上游目录适配。

## 6. 核心工作流

`.skill` 会要求 AI 智能体在每个算子任务中按以下闭环执行：

1. 任务归纳：提取算子语义、输入输出、shape、dtype、broadcast、layout、边界条件、性能要求和不支持项。
2. 仓库阅读：用 `rg` 定位相似 arrangement、application、tensor meta-operation、load/store、generated source、测试、benchmark 和示例。
3. 方案选择：基于相似代码选择 DSL 实现模式，说明为什么使用逐元素、归约、分块、stride-aware 或 fallback 方案。
4. 最小实现：只修改必要文件，保持命名、目录、异常处理、测试风格与仓库一致。
5. correctness 验证：补充与 PyTorch 或仓库参考实现对齐的测试，覆盖常规、边界、broadcast、dtype 和 layout case。
6. 性能验证：对性能敏感任务运行 benchmark 或检查 generated source/AOT build，记录基线、输入规模、命令、结果和结论。
7. 失败处理：若测试、build 或 benchmark 失败，记录现象、诊断路径、根因判断、最小修复或规避建议，再复跑验证命令。
8. 交付摘要：输出变更文件、验证命令、测试结果、性能结论、风险边界和后续建议。

## 7. 自测任务设计

| 自测任务 | 类型 | 目标 | 验证方式 | benchmark |
| --- | --- | --- | --- | --- |
| T1：broadcast gelu / bias activation | 逐元素 / 广播 | 验证 dtype、broadcast、mask 和数值容忍度处理 | 与 PyTorch 对齐，覆盖标量、尾维广播、非 2 的幂 shape | 是 |
| T2：row-wise softmax 子任务 | 归约 / 分块 | 验证 reduce 维度、tile/block、数值稳定性和尾块处理 | 与 PyTorch softmax 对齐，覆盖小尺寸和大尺寸输入 | 是 |
| T3：strided offset unary | 布局敏感 | 验证 non-contiguous、stride、offset 下 load/store 是否正确 | 构造 sliced / transposed input，与 PyTorch 参考输出对齐 | 可选 |
| T4：generated source 与 AOT build 诊断 | 性能 / 诊断 / 集成 | 对一个已有或新增算子定位 generated source、AOT build 或 benchmark 回退原因 | 记录失败现象、根因、最小修复、复跑命令和结果 | 是 |

每个自测任务都将包含：

- 输入任务说明。
- AI 智能体执行记录摘要。
- 产生的算子代码、示例、测试或修复补丁摘要。
- correctness 测试命令和结果。
- 如包含 benchmark，则记录基线、输入规模、运行命令、结果和性能结论。
- 如涉及失败诊断，则记录失败现象、根因判断、修复或规避建议和验证闭环。

## 8. Benchmark 设计

benchmark 计划遵循“可复现、可比较、可解释”的原则。

1. 基线选择：优先使用仓库已有实现或测试/benchmark 基线；如果仓库没有直接基线，则使用 PyTorch 参考或上一版实现作为对比，并说明差异。
2. 输入规模：每个 benchmark 至少覆盖小规模 correctness 友好输入和较大规模性能敏感输入；对 reduce/softmax 类任务额外覆盖不能整除 tile/block 的尾块规模。
3. 统计口径：记录 warmup、重复次数、均值/中位数或仓库 benchmark 默认统计项，避免只报告单次运行。
4. 性能结论：不仅报告快慢，还解释可能原因，例如访存模式、冗余 broadcast、stride 处理、tile/block 配置、AOT build 配置。
5. 回退阈值：若新增实现相对合理基线出现明显性能下降，将要求智能体给出 generated source 或 benchmark 层面的证据；若暂时不能修复，必须给出风险说明和 fallback。

## 9. 评测方式与量化指标

### 9.1 自测评分

自测将模拟赛题组隐藏评测协议，对每个任务按 10 分制记录：

| 子项 | 分值 | 自测判定 |
| --- | --- | --- |
| 任务完成度 | 4 | 算子语义、shape、dtype、边界条件或诊断目标是否满足 |
| 测试与验证 | 2 | 指定 correctness 测试是否通过，验证闭环是否清晰 |
| 性能意识 | 1 | 是否提供 benchmark、generated source 分析或合理优化说明 |
| 补丁最小性 | 1 | 是否无无关重构、无大面积格式化、无破坏性改动 |
| 仓库风格一致性 | 1 | 命名、结构、测试、示例和文档风格是否符合已有代码 |
| 过程与合规 | 1 | 执行记录可复查，且无密钥、无联网依赖、无绕过测试 |

### 9.2 A/B 对比

计划使用同一 AI 智能体、同一模型、同一仓库版本、同一工具权限和同一时间预算，对比“不使用 `.skill`”与“安装并触发 `.skill`”两组表现。

记录指标包括：

- 任务完成率：目标是自测 4 个任务中至少 4 个给出可运行方案，至少 3 个完整通过 correctness 验证。
- 测试通过率：目标是新增或修改的 correctness 测试全部通过；若受仓库或平台限制不能运行，必须给出可复现阻塞说明。
- 性能材料完整率：目标是至少 2 个自测任务提供完整 benchmark，至少 1 个任务提供 generated source 或 AOT build 分析。
- 迭代效率：记录从任务开始到首次测试通过的轮次、失败次数和主要修复点。
- 补丁最小性：目标是每个任务只改动相关算子、测试、示例或 benchmark 文件，不引入无关格式化。
- 过程可复查性：目标是每个任务都包含命令、结果、失败诊断和最终结论。

### 9.3 决赛目标指标

面向隐藏评测，预期目标为：

- 隐藏任务折算前原始分不低于 60/80。
- 8 个隐藏任务中至少 6 个任务的“任务完成度”达到 3/4 或以上。
- 至少 3 个隐藏任务体现有效的性能验证、generated source 分析、AOT build 分析或性能回退定位。
- 所有提交材料满足离线复现、安全合规和引用披露要求。

## 10. 预期效果

安装 `ninetooth-operator-dev` 后，预期 AI 智能体相比无 `.skill` 基线有以下改进：

1. 更快定位仓库模式：通过 repo-reading map 和 `rg` 搜索清单，减少盲目阅读文件和重复试错。
2. 更稳定完成算子闭环：从需求解析、DSL 选择、实现、测试到 benchmark 的步骤固定化，降低遗漏 dtype、layout、边界条件的概率。
3. 更强 correctness 覆盖：通过测试矩阵强制覆盖 PyTorch 对齐、broadcast、尾块、stride/offset 和数值容忍度。
4. 更明确性能意识：每个性能敏感任务都要求给出 benchmark 或 generated source/AOT build 证据，避免只追求“能跑”。
5. 更可审查的失败处理：失败案例不只给出最终补丁，还保留诊断路径、根因判断和复跑结果。
6. 更低合规风险：明确禁止密钥、联网强依赖、隐藏答案、删除测试、伪造结果和破坏性操作。

## 11. 风险与边界

### 11.1 不覆盖范围

- 不修改 NineToothed 编译器核心机制，除非赛题任务明确要求且仓库已有相似模式。
- 不承诺覆盖仓库尚未支持的 dtype、动态 shape、硬件后端或算子族。
- 不承诺对所有非连续布局都达到最优性能；会优先保证 correctness，并对性能风险做显式说明。
- 不依赖在线 LLM、云端 benchmark、私有账号、API key 或未授权数据。
- 不针对隐藏任务写硬编码规则，不包含隐藏评测答案或评测脚本规避逻辑。

### 11.2 主要风险与缓解

| 风险 | 影响 | 缓解方式 |
| --- | --- | --- |
| NineToothed 仓库接口变化 | 示例或脚本失效 | `SKILL.md` 优先要求读取当前仓库相似实现；脚本只做辅助，不硬编码路径 |
| benchmark 环境差异 | 性能结论不稳定 | 报告硬件、输入规模、重复次数和统计口径；性能结论以趋势和证据为主 |
| AI 智能体误改无关文件 | 降低补丁最小性 | 工作流要求先列修改计划，再只改算子、测试、示例、benchmark 相关文件 |
| non-contiguous 处理复杂 | correctness 或性能失败 | 提供 stride/offset 专项测试矩阵和 fallback 说明 |
| AOT build 或 generated source 工具不可用 | 性能诊断不完整 | fallback 为仓库已有测试、静态检查、最小 benchmark 或明确阻塞记录 |

## 12. 合规、依赖与引用披露

`.skill` 将遵守以下要求：

- 不包含 API key、账号凭据、私有数据、隐藏答案或个人无关信息。
- 所有脚本可离线执行，依赖使用仓库已有工具链或 Python/Bash 基础环境；如后续确需额外依赖，将在报告中说明安装方式、版本、用途和 fallback。
- 所有外部资料、参考代码、AI 辅助范围和第三方依赖将在 `REFERENCE.md` 中披露。
- 最终提交包含署名 `HONOR_CODE.md`，承诺不抄袭、不绕过评测、不伪造结果。
- 网络工具默认只用于公开资料查阅，不作为 `.skill` 运行时强依赖。

## 13. 初赛与决赛计划

| 阶段 | 截止时间 | 计划产出 |
| --- | --- | --- |
| Proposal 阶段 | 2026-05-20 | 完成 proposal，明确 `.skill` 目标、覆盖范围、自测与 benchmark 计划 |
| 初赛阶段 | 2026-06-08 | 提交 `.skill` 初版或设计草案、4 个自测任务计划、中期报告 |
| 决赛阶段 | 2026-07-13 00:00 | 提交最终 `.skill` 包、完整自测材料、最终报告、`HONOR_CODE.md`、`REFERENCE.md` |

## 14. 总结

`ninetooth-operator-dev` 将以 NineToothed 算子开发的真实工作流为中心，把“需求解析、仓库阅读、DSL 选择、最小实现、correctness 测试、benchmark、失败诊断、合规报告”组织成一个可复用 `.skill`。该方案的重点不是写一份静态说明文档，而是让固定 AI 智能体在干净仓库和统一时间预算下更稳定地交付可测试、可复现、具备性能意识且符合仓库风格的九齿算子实现。
