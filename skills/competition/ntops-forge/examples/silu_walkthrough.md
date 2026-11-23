# 完整示例：用 ntops-forge 完成 silu 算子

本示例展示 Agent 加载 `ntops-forge` 后，从规格到 GPU pytest 通过的全流程（与大赛 T3-1-1 自测要求对齐）。

## 1. 任务说明

- **算子**：silu（逐元素一元）
- **数学**：`silu(x) = x * sigmoid(x)`
- **验收**：`tests/test_silu.py` 全部通过 + `compare_ref` 与官方内核一致

## 2. Agent 执行步骤

```bash
source /root/miniconda3/bin/activate base
cd /root/work/skill
pip install -q pytest && pip install -q -e /root/work/ntops

# PLAN：读取 spec
python scripts/forge.py spec silu

# 单算子五段流水线
python scripts/forge.py run silu --ntops-root /root/work/ntops
```

## 3. 各阶段预期输出

| 阶段 | 关键输出 |
|------|----------|
| PLAN | `[PLAN] OK: plan ready` |
| CODEGEN | `Wrote /tmp/silu_forge_kernel.py` |
| GUARD | `OK: application() matches reference` |
| PROVE | `8 passed` |
| SHIP | `FORGE OK (silu) in ~7s` |

## 4. 产生物摘要

- 内核：`/tmp/silu_forge_kernel.py`（formula 自动注入）
- 审计：`docs/forge_runs.jsonl`（五阶段 `ok: true`）
- PR 提示：分支 `2026-spring-hongwei-2026-T3-1-1`

## 5. 无 skill 基线对比

```bash
python scripts/run_baseline_demo.py --reset-csv --ops silu
python scripts/record_run.py --mode treatment --task silu \
  --preflight-pass --pytest-pass --steps 1 --interventions 0 --elapsed 7
python scripts/eval_ab.py --input docs/ab_runs.csv --output docs/AB_Report.md
```

Baseline：preflight 0%、6 步、4 次人工介入；Treatment：preflight 100%、1 步、0 介入。

## 6. 失败时

```bash
python scripts/forge.py diagnose --log docs/forge_runs.jsonl
```

对照 `skills/ntops-forge/fix_cards.md` 中的 FC-xxx 修复卡。
