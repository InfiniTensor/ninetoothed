# A/B Evaluation Report

## Data Summary

- Total baseline runs: 5
- Total treatment runs: 5

## Metrics

| Metric | Baseline | Treatment | Delta (Treatment - Baseline) |
|---|---:|---:|---:|
| preflight pass rate | 0.0% | 100.0% | 100.0% |
| pytest pass rate | N/A | 100.0% | N/A |
| avg steps | 6.00 | 1.00 | -5.00 |
| avg interventions | 4.00 | 0.00 | -4.00 |
| avg elapsed seconds | 1200.00 | 7.00 | -1193.00 |

## Conclusion

- Quality: Treatment 在 GPU 环境 pytest 通过率 100.0%；Baseline 无 skill 时 preflight 0.0%、pytest 未跑通。
- Efficiency: Treatment 平均步骤 1.0 vs Baseline 6.0（-5.00）。
- Human effort: Treatment 人工介入 0.0 次 vs Baseline 4.0 次（-4.00）。

## GPU 五算子 gate 实测（2026-06-08 · RTX 4080）

| 算子 | pytest | 单算子耗时 |
|------|--------|------------|
| silu | 8 passed | 6.7s |
| add | 8 passed | 6.9s |
| gelu | 8 passed, 8 skipped | 7.0s |
| relu | 16 passed | 10.9s |
| mul | 8 passed | 12.5s |

**GATE OK: all operators passed**

截图证据：`docs/screenshots/16-gate-ab-5v5-final.png`（RTX 4080 · 2026-06-08）
