# Performance Regression Case — 示例场景

## 目标

模拟和诊断性能退化的场景，作为 "性能基准 + 调试" 流程的完整展示。

## 场景设计

在 Matmul 2D 模式下实现一个矩阵乘法 kernel，然后引入一个常见性能问题（如 block size 过小，或 load/store 模式不高效），通过 benchmark 对比发现问题。

## 流程

1. **基线实现** — 使用默认的 matmul_2d 模式（BLOCK_SIZE=128）
2. **退化引入** — 切换 BLOCK_SIZE=16（过小的 tile 导致利用率下降）
3. **诊断** — benchmark 对比，定位性能下降
4. **修复** — 调整参数或 autotune，恢复性能

## 关键观察

| 配置 | 预期性能 | 原因 |
|------|----------|------|
| BLOCK_SIZE=128 | ✅ 高 | 充分利用 SM |
| BLOCK_SIZE=16 | ❌ 低 | tile 过小，load/store 开销大 |
| Autotune | ✅ 最高 | Heuristic 选择最优参数 |

## 运行

```bash
python examples/performance_regression_case/run.py         # verify correctness
python examples/performance_regression_case/benchmark.py   # compare configs
```

## 预期输出

Benchmark 应清晰显示 BLOCK_SIZE=16 的严重退化，以及 autotune 恢复性能的效果。
