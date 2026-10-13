# 任务说明：官方 ntops 性能对比

在干净的官方 `ntops` 仓库上运行一组代表性算子，与 PyTorch eager 做同卡 benchmark。

目的：

- 检查本 skill 记录的性能差距是否只来自本地新增算子。
- 验证官方仓库已有算子在当前环境和输入下的 PyTorch 对比结果。
- 给中期报告提供更清楚的性能边界。

命令：

```bash
cd <clean-ntops-repository>
python3 \
  <ntskill-repository>/skills/competition/ntops-dev/scripts/run_official_ntops_benchmark.py \
  --warmup 20 --iters 100
```

要求：

- 使用干净官方 `ntops` 仓库，不使用本项目新增的 `maximum`、`square`。
- 记录 GPU、PyTorch/CUDA 版本、warmup、iters、shape、dtype、ntops 延迟、PyTorch 延迟和相对比值。
- 结论只限定到本次算子、输入和环境，不外推到全部 ntops 算子。
