# ST1：逐元素 / 广播类算子（已完成）

**赛题 4.2 类型**：逐元素或广播类（add、relu、gelu 等）  
**状态**：✅ RTX 4080 GPU 实测通过

## 任务说明

在 ntops 仓库为 5 个 elementwise 算子完成：规格解读 → 内核生成 → 双护栏 → pytest → 审计。

| 算子 | family | pytest |
|------|--------|--------|
| silu | elementwise_unary | tests/test_silu.py |
| add | elementwise_binary | tests/test_add.py |
| gelu | elementwise_unary | tests/test_gelu.py |
| relu | elementwise_unary | tests/test_relu.py |
| mul | elementwise_binary | tests/test_mul.py |

## Agent 执行记录摘要

```bash
python scripts/forge.py gate --ntops-root /root/work/ntops
```

- 每算子 1 条流水线，PLAN→SHIP 五阶段全 `ok`
- 单算子约 7s；五算子合计约 35–44s
- 人工介入 0 次（gate 一键）

## 产物摘要

- 内核：`/tmp/*_forge_kernel.py`（formula 注入）
- 审计：`docs/forge_runs.jsonl`
- A/B：`docs/ab_runs.csv`（5 baseline / 5 treatment）

## Correctness 验证

```bash
python scripts/forge.py gate --ntops-root /root/work/ntops
# 每算子：matches reference + pytest passed
```

| 算子 | GUARD | pytest |
|------|-------|--------|
| silu | ✅ | 8 passed |
| add | ✅ | 8 passed |
| gelu | ✅ | 8 passed |
| relu | ✅ | 16 passed |
| mul | ✅ | 8 passed |

## 证据

- 截图：`docs/screenshots/16-gate-ab-5v5-final.png`
- 示例：`skills/ntops-forge/examples/silu_walkthrough.md`
