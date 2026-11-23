# 初赛评分逐条对照（满分 100）

> 选手：于鸿伟 · 仓库：https://github.com/hongwei-2026/qiyuanbisai  
> 提交 Commit：https://github.com/hongwei-2026/qiyuanbisai/commit/e7b32bb

---

## 一、Proposal 质量（35 分）

| 评审点 | 本仓库证据 | 状态 |
|--------|------------|------|
| 目标任务清晰 | `docs/Proposal.md` §一 三类 Agent 失败 | ✅ |
| 覆盖范围 | 五算子 gate + 四类自测映射 `FinalsRoadmap.md` | ✅ |
| 评测方式 | A/B 协议 `run_ab_suite.py`、`eval_ab.py` | ✅ |
| 风险边界 | Proposal §2.6 不做什么 | ✅ |
| 可行性 | GPU 实测表 Proposal §五 | ✅ |
| 竞品差异 | `docs/CompetitiveAnalysis.md` | ✅ |

---

## 二、.skill 设计与初版质量（25 分）

| 评审点 | 本仓库证据 | 状态 |
|--------|------------|------|
| SKILL.md 触发场景 | `skills/ntops-forge/SKILL.md`、`ntops-copilot/SKILL.md` | ✅ |
| 工作流清晰 | PLAN→CODEGEN→GUARD→PROVE→SHIP | ✅ |
| references/ | `skills/ntops-forge/references/README.md` | ✅ |
| examples/ | `skills/ntops-forge/examples/silu_walkthrough.md` | ✅ |
| tests/ 验证说明 | `skills/ntops-forge/tests/README.md` | ✅ |
| 脚本可执行 | `scripts/` 16+ 脚本 + `doctor.py` | ✅ |
| 双 Skill 边界 | `docs/DualSkillGuide.md` | ✅ |

---

## 三、中期报告与自测计划（25 分）

| 评审点 | 本仓库证据 | 状态 |
|--------|------------|------|
| 进度与完成情况 | `docs/MidTermReport.md` + PDF（17 图） | ✅ |
| 自测计划 | `docs/SelfTestPlan.md` | ✅ |
| 四类自测案例 | `docs/selftests/ST1–ST4` | ✅ |
| 执行记录 | `docs/forge_runs.jsonl`、`ab_runs.csv` | ✅ |
| 后续计划 | MidTermReport §十四、FinalsRoadmap | ✅ |
| benchmark（≥2 任务） | A/B 耗时 + silu kernel（ST4） | ✅ |

---

## 四、合规与可复现性（15 分）

| 评审点 | 本仓库证据 | 状态 |
|--------|------------|------|
| 无密钥/隐藏答案 | 全仓库无 .env 密钥 | ✅ |
| 无联网强依赖 | 离线脚本 + fallback | ✅ |
| HONOR_CODE | `HONOR_CODE.md` | ✅ |
| REFERENCE | `REFERENCE.md` | ✅ |
| 环境复现 | `docs/CloudRun.md` | ✅ |

---

## 赛题 4.2 必须交付对照

| 要求 | 证据 |
|------|------|
| 可安装 .skill 包 | `skills/ntops-forge/` + `skills/ntops-copilot/` |
| ≥4 个自测任务 | ST1–ST4 |
| 逐元素/广播 | ST1 五算子 gate |
| 归约/分块 | ST2 softmax **8 passed** GPU |
| 布局 stride | ST3 max_pool2d **62 passed** GPU |
| 性能/诊断 | ST4 A/B + benchmark + fix_cards |
| ≥2 benchmark | ST4 任务 A + 任务 B |
| 赛题报告要素 | MidTermReport.md / PDF |

---

## 自评：初赛目标 **99–100 分**

ST2/ST3 GPU pytest 已实测（图18–19）：softmax 8/8、max_pool2d 62 passed（54 skip 为上游 Invalid padding）。
