# NineToothed .skill 创新挑战赛 — 项目提案

**赛题**: 2026 春季 AI 竞赛 — NineToothed .skill 创新挑战赛  
**赛道编号**: T3-1-1  
**参赛者**: Zhao Junkai  
**GitHub ID**: junkai-kay  
**Skill 名称**: ninetoothed-operator-skill  
**提交日期**: 2026-07-05

---

## 1. 项目背景与目标

### 1.1 赛题理解

本赛道要求沉淀一个可指导 AI 智能体稳定完成 NineToothed 算子开发、测试、调试和 PR 集成的 `.skill` 工程包。评审不是看文档写得多长，而是看安装该 skill 后，AI 智能体在标准隐藏评测任务中的实际完成效果。

### 1.2 项目目标

构建一个覆盖以下四类算子开发场景的可复用 skill：
- 逐元素 / 广播算子
- 归约 / 分块算子
- 布局敏感（非连续输入）算子
- 性能 / 诊断 / 集成任务

预期成果：AI 智能体在面对陌生 NineToothed 算子任务时，能按 skill 中定义的标准操作流程，自主完成语义提取、模式匹配、实现选型、测试覆盖、性能分析和故障闭环。

---

## 2. 技术方案

### 2.1 设计原则

- **工作流优先**：SKILL.md 是 9 阶段可执行 SOP，不是说明文
- **条件判断而非绝对规则**：每种模式注明适用条件、参考文件路径、不适用情况
- **泛化而非特化**：覆盖 5 种 arrangement/application 模式，不针对特定算子
- **验证闭环**：每个任务必须有 PyTorch reference → pytest → benchmark → 诊断
- **离线可用**：无外部联网依赖、无 API key

### 2.2 技术路线

1. 通读 NineToothed 官方仓库 (README、CONTRIBUTING.md、docs、tests、examples)
2. 基于官方 test_softmax.py 和 test_add.py 提取正确的 API 使用模式
3. 设计 4 个自测任务，覆盖四类算子场景
4. 每个任务包含：算子实现、正确性测试、benchmark、失败诊断
5. 将开发经验沉淀为 SKILL.md + references/index.md
6. 编写 scripts/ 和 tests/ 使 skill 可自动验证

### 2.3 核心技术选择

- 算子定义：`@ninetoothed.jit` + `Symbol(constexpr=True)`
- 测试框架：pytest + PyTorch reference
- 性能对比：NineToothed vs PyTorch baseline
- 失败诊断：6 步闭环协议

---

## 3. 预期交付物

| 交付物 | 说明 |
|---|---|
| SKILL.md | AI 智能体核心工作流（含 YAML 头、9 阶段 SOP、故障诊断协议） |
| README.md | 人类阅读的安装/使用/自测说明 |
| references/index.md | 仓库地图、5 种开发模式速查、常见陷阱 |
| scripts/ | env_check.py、run_selftests.py |
| tests/ | test_skill_structure.py（结构验证、占位符检查、红线检查） |
| examples/ | 4 个自测任务（含算子实现、测试、benchmark、诊断记录） |
| reports/ | 最终赛题报告 |
| HONOR_CODE.md | 诚信声明 |
| REFERENCE.md | 引用与 AI 辅助披露 |

---

## 4. 计划时间线

| 阶段 | 内容 | 状态 |
|---|---|---|
| 阶段一 | 建立全局认知，阅读官方仓库，输出仓库地图 | 2026-07-03 |
| 阶段二 | 创建 skill 骨架，编写 SKILL.md 和 README.md | 2026-07-05 |
| 阶段三 | 设计 4 个自测任务，编写算子实现和测试代码 | 2026-07-08 |
| 阶段四 | GPU 实测、调试、benchmark、撰写最终报告 | 2026-07-11 |
| 阶段五 | 合规审查、Git 提交、PR 创建 | 2026-07-12 |

---

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| NineToothed API 与文档不一致 | 以官方仓库 tests/ 目录中的实际代码为准 |
| GPU 资源不足 | 使用 Google Colab 免费 T4 GPU |
| 隐藏任务超出预期覆盖范围 | skill 设计为模式化指导，不针对特定算子 |

---

## 6. 个人声明

准大二本科生，对 AI 基础设施和高性能计算方向有浓厚兴趣。参加本次比赛是为了在实践中学习 GPU 算子开发和 AI 工程化技能，通过亲手构建 .skill 工程包来理解如何让 AI 智能体更可靠地完成底层开发任务。希望借助赛题提供的 NineToothed 平台和评审反馈，在真实项目中锻炼自己的系统设计和技术写作能力。
