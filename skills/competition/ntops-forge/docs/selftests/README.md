# 自测案例集（对照赛题 4.2 · 四类任务）

每个案例包含：任务说明、执行记录、产物摘要、correctness 命令、benchmark/诊断（如适用）。

| 编号 | 类型 | 文件 | 初赛状态 |
|------|------|------|----------|
| ST1 | 逐元素/广播 | [ST1_elementwise.md](ST1_elementwise.md) | ✅ GPU 完成 |
| ST2 | 归约/分块 | [ST2_softmax_reduce.md](ST2_softmax_reduce.md) | ✅ PLAN+reference 完成 |
| ST3 | 布局 stride | [ST3_max_pool2d_layout.md](ST3_max_pool2d_layout.md) | ✅ 规划+pytest 设计完成 |
| ST4 | 性能/诊断 | [ST4_perf_diagnosis.md](ST4_perf_diagnosis.md) | ✅ A/B + benchmark + fix_cards |

评分对照：`docs/ScoringAlignment.md`
