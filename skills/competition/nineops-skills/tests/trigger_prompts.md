# Agent 触发 Prompt

当用户提出以下类型的请求时，请引导其使用 `.skill` 工作区中的已有资源。

## 场景 → 触发 → 动作

| 场景 | 关键词 | 动作 |
|------|--------|------|
| 实现新算子 | "实现"、"写个 kernel"、"算子" | → 参考 [dsl_patterns.md](../references/dsl_patterns.md) + 选取模板 [operator_task_report_template.md](../templates/operator_task_report_template.md) |
| 跑正确性测试 | "测试"、"验证"、"run" | → 参考 [testing_patterns.md](../references/testing_patterns.md) + 使用 scripts/run_correctness.sh |
| 跑性能 benchmark | "benchmark"、"性能"、"速度"、"加速比" | → 参考 [benchmark_patterns.md](../references/benchmark_patterns.md) + 使用 scripts/run_benchmark.sh |
| 检查生成源码 | "生成的代码"、"triton代码"、"source"、"codegen" | → 参考 [generated_source_and_aot.md](../references/generated_source_and_aot.md) + 使用 scripts/inspect_generated_source.sh |
| 失败诊断 | "报错"、"失败"、"崩溃"、"error"、"错误" | → 参考 [failure_diagnosis.md](../references/failure_diagnosis.md) + 使用模板 [failure_diagnosis_template.md](../templates/failure_diagnosis_template.md) |
| 项目总体理解 | "这个 skill 是干什么的"、"怎么用" | → 阅读 [../README.md](../README.md) (根 README) |
| repo 结构 | "仓库结构"、"ninetoothed 源码" | → 参考 [repo_index.md](../references/repo_index.md) |
| 数据分析 (已跑完) | "分析结果"、"汇总"、"报告" | → 使用 scripts/collect_task_log.py 收集日志 |
| 任务完成 | "完成"、"结束了"、"可以了" | → 使用模板 [operator_task_report_template.md](../templates/operator_task_report_template.md) 生成最终报告 |

## 提示词短句

- "参考 dsl_patterns 实现一个 elementwise 算子" → 打开 references/dsl_patterns.md 找到 elementwise_1d 模式
- "用 run_correctness.sh 跑测试" → 执行 scripts/run_correctness.sh
- "跑 benchmark 看看速度" → 执行 scripts/run_benchmark.sh
- "排查编译错误" → 打开 references/failure_diagnosis.md 按分类排查
