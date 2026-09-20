# 同步上游整数类型别名（2026-09-20）

将官方#218（7ffdab8a9ae5a1da03ec2e7359a2199a4475a142）合入功能分支3011b3f，得到双父合并提交8c0c9bbc796ade27fb84fdb62b2fdeb40e666936，受测树为01becd0e737d49b2661ccda55c087e4588a2819c。Git三方合并无冲突，共8个上游变更文件。compiler/passes.py同时保留本项目的ProvenancePass、record_pass、seed_origins和上游normalize_dtype更新，未覆盖解释器的追踪改动。

## 回归

NumPy-only环境（无Torch/Triton）运行统一CPU回归：

```text
799 passed, 15 deselected in 28.84s
```

隔离Torch CPU环境运行新增别名、PyTorch CPU适配、默认pass和dtype缓存回归：

```text
96 passed, 1 deselected, 1 warning in 2.39s
```

96项包含上游新增的全部40项类型别名测试。唯一排除项是test_cuda_tensor_is_rejected_by_cpu_interpreter，需要实际CUDA张量，本地Mac未运行。唯一警告来自Torch稀疏张量默认invariant检查提示，对应的稀疏输入拒绝断言通过。两套测试有重叠，不相加为895个独立测试。

Ruff、171文件格式检查、项目贡献风格和git diff --check通过；以当前源码及完整依赖执行Sphinx -W严格构建通过。早先文档构建误用旧安装路径且缺少Torch依赖的日志独立保留，不是新代码的功能失败。205个冻结输入在数值回归前后SHA一致，源码与完整增量可由合并提交恢复。

## 实机证据范围

本次是上游兼容性同步，没有重新租用GPU或执行新的A100测试。2026-09-19的A100全库1206通过/2多卡跳过仍严格对应a86bea9；该历史结果不能当作8c0c9bbc796ade27fb84fdb62b2fdeb40e666936组合的新实机验收。同步后的CPU和类型别名结果单独记录，后续上游工作流仍须维护者批准，PR合并仍须维护者评审。

完整日志、环境、JUnit、源码SHA和三方合并补丁：[固定证据](https://github.com/a962695448-rgb/ninetoothed/tree/e12afe25929a890c073fc85a0d154599ec92f253/docs/validation/upstream-sync-20260920)。
