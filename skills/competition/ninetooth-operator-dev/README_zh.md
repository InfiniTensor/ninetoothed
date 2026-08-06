# NineToothed 算子开发 Skill

[English](README.md) | 中文

## 项目概览

本目录用于存放 NineToothed `.skill` 创新挑战 T3-1-1 方向的开箱即用 AI-agent skill、最终交付材料与 proposal，位于 `skills/competition/ninetooth-operator-dev/`。当前交付包含 skill 框架、四个自测任务规格、机械校验脚本、最终报告、Honor Code 和引用披露。

自测任务中的 correctness 或 benchmark 结果如果尚未真实执行，会明确保留 `Not run yet` 或 blocker 字段，不伪造结果。

该 proposal 面向 AI 智能体完成算子开发任务的真实工作流，重点关注需求理解、仓库阅读、DSL 实现、correctness 测试、benchmark、generated source / AOT build 分析以及失败诊断。

## 快速开始

把本目录当作 NineToothed checkout 内的 skill 包和使用说明。

1. 克隆 NineToothed 并进入 checkout：

```bash
git clone https://github.com/InfiniTensor/ninetoothed.git
cd ninetoothed
export NINETOOTHED_REPO="$PWD"
export NINETOOTHED_SKILL_DIR="$NINETOOTHED_REPO/skills/competition/ninetooth-operator-dev"
```

2. 把 skill 暴露给你的智能体运行环境。以 Codex 本地 skill 目录为例：

```bash
mkdir -p ~/.codex/skills
cp -R "$NINETOOTHED_SKILL_DIR" ~/.codex/skills/
```

如果你的智能体使用其他 skill 目录，就把
`$NINETOOTHED_SKILL_DIR` 复制到对应位置。也可以不复制，直接让智能体读取 NineToothed checkout 里的这个 skill 目录。

如果运行环境不支持命名 skill，可以让智能体直接从
`skills/competition/ninetooth-operator-dev/SKILL.md` 开始读取。

3. 如需真实运行测试或 benchmark，准备 NineToothed Python 环境：

```bash
cd "$NINETOOTHED_REPO"
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
```

`ninetoothed` 要求 Python 3.10 或更高版本。GPU correctness 和 benchmark 还依赖匹配的 PyTorch、Triton、CUDA 和设备环境。

4. 验证本 skill 包结构：

```bash
cd "$NINETOOTHED_REPO"
python "$NINETOOTHED_SKILL_DIR/scripts/lint_skill_structure.py" "$NINETOOTHED_SKILL_DIR"
python "$NINETOOTHED_SKILL_DIR/tests/test_structure.py"
python "$NINETOOTHED_SKILL_DIR/tests/test_submission_package.py"
```

5. 让智能体使用 skill。示例提示词：

```text
Use $ninetooth-operator-dev.
Upstream NineToothed checkout: ${NINETOOTHED_REPO}.
Task: implement a bias + ReLU operator with PyTorch-aligned correctness tests.
Read the skill entrypoint first, then load only the references needed for this task.
```

实际使用时始终把 `NINETOOTHED_REPO` 设置为当前 NineToothed checkout 根目录。

## 背景说明

AI 智能体已经可以完成许多通用编码任务，但算子开发往往涉及更多工程细节：数学语义、shape 规则、dtype 约束、broadcast、非连续布局、load/store、generated source、测试矩阵、性能回退和构建问题。

本项目把这些容易分散在仓库和开发经验中的知识整理成可复用的 `.skill` 工作流，并把关键地图、检查器、自测任务规格和最终交付说明留在仓库中，让 AI 智能体在统一环境和时间预算下更稳定地完成任务。

## Skill

Skill 名称：

```text
ninetooth-operator-dev
```

该 skill 指导 AI 智能体完成一个完整的算子开发闭环：

- 理解算子需求；
- 阅读仓库已有实现模式；
- 选择合适的 DSL 表达方式；
- 实现最小必要补丁；
- 编写 correctness 测试；
- 在需要时补充 benchmark 或 generated source 检查；
- 诊断失败并记录验证结果。

## 项目目标

### 目标一：提升算子实现稳定性

帮助 AI 智能体写出更符合语义、shape、dtype、layout 和边界条件要求的算子实现。

### 目标二：强化测试闭环

让每个算子任务都包含明确的 correctness 验证，并尽量与 PyTorch 参考实现或仓库已有实现对齐。

### 目标三：引入性能意识

将 benchmark、generated source 检查、AOT build 检查和性能回退分析纳入常规工作流。

### 目标四：保证可复现评测

使 `.skill` 设计和最终交付材料能够在干净仓库、离线环境和统一工具权限下复查。

### 目标五：规范失败诊断

要求智能体记录失败现象、诊断路径、根因判断、最小修复和复跑结果。

## 覆盖场景

### 逐元素与广播算子

可覆盖 activation、标量算子、broadcast binary operator、mask-aware operator 等任务。

### 归约与分块算子

可覆盖 reduce、softmax 子任务、pooling 子任务、tile/block 计算等任务。

### 布局敏感算子

可覆盖 non-contiguous input、stride、offset、slice input、contiguous fallback 等场景。

### 性能与集成任务

可覆盖 benchmark 补齐、generated source 检查、AOT build 配置、性能回退定位和最小集成修复。

## 评测方式

proposal 计划按照赛题规则对齐以下维度：

- 任务完成度；
- correctness 测试与验证；
- 性能意识；
- 补丁最小性；
- 仓库风格一致性；
- 过程记录与合规性。

## 自测材料

当前包含四类自测任务规格：

1. `examples/elementwise-broadcast/TASK.md`
2. `examples/reduction-block/TASK.md`
3. `examples/layout-sensitive/TASK.md`
4. `examples/performance-diagnostics/TASK.md`

每个自测任务包含任务输入、智能体执行摘要、补丁摘要、correctness 命令、结果字段、benchmark 命令或 benchmark 范围，以及 unsupported scope 说明。

## Benchmark 计划

benchmark 材料记录：

- baseline 实现；
- 输入 shape 与 dtype；
- warmup 和重复次数；
- benchmark 运行命令；
- 结果数据；
- 性能差异解释；
- 已知限制。

## 预期效果

预期该 skill 可以帮助 AI 智能体提高隐藏任务完成率，减少遗漏边界条件的情况，强化测试覆盖，增强性能分析意识，并输出更清晰、可审查的任务报告。

## 包内容

```text
skills/competition/ninetooth-operator-dev/
├── FINAL_REPORT.md           # 最终交付报告
├── HONOR_CODE.md             # Honor Code 声明
├── README.md                 # 英文项目概览
├── README_zh.md              # 中文项目概览
├── REFERENCE.md              # 引用与披露
├── SKILL.md
├── proposal.md               # 完整 proposal 文档
├── agents/openai.yaml
├── references/
├── scripts/
├── examples/
├── subagent-sessions/
└── tests/
```

## NineToothed Checkout

本 skill 位于 NineToothed 上游仓库内部。请把 `NINETOOTHED_REPO` 设置为
仓库根目录。skill 中的上游文件引用都相对于这个 checkout，例如
`${NINETOOTHED_REPO}/tests/test_add.py`。

## 合规说明

项目不应包含隐藏评测答案、API key、账号凭据、私有数据或绕过测试的脚本。外部资料、生成式 AI 辅助范围和本地规则来源处理已在 `REFERENCE.md` 中披露；Honor Code 声明位于 `HONOR_CODE.md`。

## 当前状态

本目录聚焦于可复用的 NineToothed 算子开发 skill 及其提交材料。独立开发
历史位于 `https://github.com/LaiQuan-conquer/NineToothed-OperatorSkills`。
