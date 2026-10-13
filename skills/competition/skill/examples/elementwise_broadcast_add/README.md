# Element-wise Broadcast Add — 示例场景

## 目标

实现一个支持广播的 element-wise add kernel，作为 `elementwise_1d` 模式的展示。

## 任务描述

实现 `ninetoothed_add` 函数，接受两个 tensor，支持广播语义，使用 `elementwise_1d` 模板。

## 关键点

1. **Broadcast** — 第二个输入可以比第一个输入小，需要自动 broadcast
2. **Mask 处理** — 边界处理
3. **BLOCK_SIZE** — 使用 1024 或 autotune

## 运行

```bash
python examples/elementwise_broadcast_add/run.py        # 测试正确性
python examples/elementwise_broadcast_add/benchmark.py  # 基准测试
```

## 预期结果

- 所有 correctness 测试通过（fp32/fp16, 各种 shape, 各种 broadcast 场景）
- Benchmark 至少达到 PyTorch CUDA 80% 的性能

## Torture Test 提示

- scalar broadcast: `y.shape=(1,)` with `x.shape=(N,)`
- strided broadcast: `x.shape=(128, 64)`, `y.shape=(64,)` broadcasting across dim=-1
- non-contiguous: `y = y.as_strided(...)` with non-standard strides
