# Selftest Tasks — Agent 自我校验任务

这些任务用来验证 `.skill` 工作区的完整性和 agent 能力。每个任务应能在 **不访问外部网络** 的情况下，仅基于 `.skill` 内部文档完成。

## 任务 1: 实现一个 elementwise 加法 kernel

- 打开 [dsl_patterns.md](../references/dsl_patterns.md) 找到 elementwise_1d 模式
- 使用 `Tensor` + `Symbol` + 1D arrangement + `application(ntl.add)` + `make`
- 验证: 输入 a(T), b(T) → c(T)，结果应与 a+b 一致
- 参考已有示例: [elementwise_broadcast_add](../examples/elementwise_broadcast_add/run.py)
- 预期耗时: 手动实现 ≤15 min

## 任务 2: 测试覆盖度检查

- 打开 [testing_patterns.md](../references/testing_patterns.md) 检查 4 个维度 (dtype, shape, broadcast, stride)
- 对任务 1 实现的加法 kernel，编写 pytest 参数化测试覆盖至少 6 个组合
- 验证: pytest --verbose 全部通过
- 预期耗时: ≤20 min

## 任务 3: AOT 编译 / 查看生成源码

- 对任务1的加法 kernel 调用 scripts/inspect_generated_source.sh 查看生成 Triton IR
- 确认 kernel name、参数签名、loop structure 与预期一致
- 预期耗时: ≤10 min

## 任务 4: 性能 Benchmark 分析

- 对任务1的加法 kernel，编写 benchmark 对比输入规模: (1024,), (4096,), (16384,)
- 使用 scripts/run_benchmark.sh 执行并保存结果
- 观察: 是否存在某个规模下 kernel 慢于 PyTorch（当数据量小或带宽受限时可能出现）
- 预期耗时: ≤20 min

## 任务 5: 识别一个性能退化

- 打开 [performance_regression_case](../examples/performance_regression_case/) 的 benchmark.py
- 运行 `python examples/performance_regression_case/benchmark.py`
- 观察: BLOCK_SIZE=16 比 BLOCK_SIZE=128 慢多少倍
- 预期耗时: ≤30 min (含编译)

## 任务 6: 非连续 stride 处理

- 打开 [non_contiguous_stride_case](../examples/non_contiguous_stride_case/run.py)
- 理解 10 种 stride 变体的测试方法
- 对任务1的加法 kernel 增加非连续测试
- 预期耗时: ≤15 min

## 任务 7: 使用模板报告

- 打开 [operator_task_report_template.md](../templates/operator_task_report_template.md)
- 填写任务1～4的完整报告
- 预期耗时: ≤10 min

## 任务 8: 故障注入诊断

- 人为修改加法 kernel 的 BLOCK_SIZE 为 0
- 运行新 kernel，观察报错信息
- 打开 [failure_diagnosis.md](../references/failure_diagnosis.md) 按诊断流程解决
- 使用 [failure_diagnosis_template.md](../templates/failure_diagnosis_template.md) 记录故障
- 预期耗时: ≤10 min
