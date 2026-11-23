---
name: ntops-forge
description: >-
  NineToothed ntops operator FACTORY skill. Five-stage pipeline PLAN→CODEGEN→GUARD→PROVE→SHIP,
  taxonomy routing, failure fix-cards, forge audit log. Use instead of ad-hoc scripts when building
  ntops kernels for 2026-spring contest. 九齿算子工厂, premake, element_wise, pytest, InfiniTensor.
---

# ntops-forge — 九齿算子工厂

**定位**：`ntops-copilot` 是「副驾驶」；`ntops-forge` 是「流水线工厂」。

Agent 不再拼凑零散命令，而是执行 **五段流水线**，每段有明确输入/输出与失败路由。

## 包结构（符合 Agent Skills 建议）

| 目录 | 内容 |
|------|------|
| `SKILL.md` | 本文件：触发场景与工作流 |
| `specs/` | 算子规格（含决赛规划 softmax/max_pool2d） |
| `examples/` | 完整示例 `silu_walkthrough.md` |
| `references/` | 查阅索引（taxonomy、官方文档） |
| `tests/` | skill 有效性验证说明 |
| `scripts/README.md` | 可执行脚本索引（仓库 `scripts/`） |
| `taxonomy.md` / `fix_cards.md` | 路由与失败诊断 |

## 与 copilot 的差异

| | ntops-copilot | ntops-forge |
|---|---------------|-------------|
| 心智模型 | 辅助 + 脚本集合 | **工厂流水线** |
| 入口 | `run_task.py` | `forge.py run <op>` |
| 规格 | task YAML | **forge spec**（含 family/taxonomy） |
| 失败 | 人工查文档 | **`fix_cards` 自动诊断** |
| 审计 | CSV 可选 | **jsonl 全链路日志** |

## 30 秒上手

```bash
source /root/miniconda3/bin/activate base   # GPU 云
cd /root/work/skill

# 列出可锻造算子
python scripts/forge.py list

# 一键跑完整流水线（推荐）
python scripts/forge.py run silu --ntops-root /root/work/ntops

# 演示闸门：silu + add + gelu + relu + mul 连续验收
python scripts/forge.py gate --ntops-root /root/work/ntops

# A/B 一键（baseline + gate + 报告）
python scripts/run_ab_suite.py --ntops-root /root/work/ntops

# 失败时诊断
python scripts/forge.py diagnose --log docs/forge_runs.jsonl

# 从一句话生成 spec（创新演示）
python scripts/forge_spec.py "add binary x plus y" --out skills/ntops-forge/specs/custom_add.yaml
```

## 五段流水线

```
PLAN ──► CODEGEN ──► GUARD ──► PROVE ──► SHIP
 │         │          │         │         │
规格解读   生成内核    结构+语义   pytest    PR材料+审计
taxonomy   +torch     preflight   精准文件   jsonl日志
```

### PLAN

读 `specs/<op>.yaml`，按 `taxonomy.md` 判定 family/pattern，输出执行计划。  
复杂算子（norm/attention）在此阶段 **停止** 并提示先读 reference。

### CODEGEN

`scaffold_kernel`（注入 formula）+ `scaffold_torch`；可选 `--register`。

### GUARD

`preflight --strict` + `compare_ref`（若 spec 有 reference）。

### PROVE

`pytest` 仅跑 spec 中的 `pytest` 文件（**禁止全量**）。

### SHIP

打印 PR 分支/标题，写入 `docs/forge_runs.jsonl`，可选 `record_run` 进 A/B CSV。

## 硬性原则

1. **只跑 spec 指定的 pytest 文件**
2. **GUARD 不过禁止 SHIP**
3. **禁止 Triton 交卷**
4. **工作目录**：`/root/work/skill`（不是 `$HOME`）

## 资源

| 文件 | 作用 |
|------|------|
| `specs/*.yaml` | 工厂规格（比 task 卡更丰富） |
| `taxonomy.md` | 算子分类路由 |
| `fix_cards.md` | 失败诊断卡 |
| `../ntops-copilot/formulas.md` | 公式速查（复用） |

## 大赛 PR

- 分支：`2026-spring-<github_id>-<contest_id>`
- 标题：`[2026春季][<contest_id>] <github_id>`
- SHIP 阶段自动打印

