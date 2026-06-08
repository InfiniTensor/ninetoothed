# Benchmark 设计规范

## 设计原则

Benchmark 的设计目标是对齐以下 8 个对照要素：

1. **算子** — 被测试的具体 kernel
2. **输入规模** — shape, dtype, broadcast 配置
3. **硬件信息** — GPU 型号、CUDA 版本
4. **布局配置** — tile size, block size, autotune 参数
5. **Baseline** — PyTorch CUDA 实现（或其他 baseline）
6. **Ninetoothed** — 被测 DSL 生成 kernel
7. **实现差异** — load/store 模式、tile/block 布局选择
8. **Fallback 情况** — generated source 或 AOT build 中的回退路径

## Benchmark 输入规模矩阵

| 规模等级 | 典型值 | 说明 |
|----------|--------|------|
| S (small) | (128,), (64, 64) | 测试 kernel launch 开销 |
| M (medium) | (4096,), (256, 768) | 典型推理/训练场景 |
| L (large) | (65536,), (1024, 1024) | 压力测试 |
| XL (extra) | (131072,), (4096, 4096) | 极限规模 |

## 测量指标

| 指标 | 单位 | 说明 |
|------|------|------|
| latency | ms | 单次调用延迟（预热后均值） |
| throughput | GB/s | 有效带宽 |
| TFLOPS | TF/s | 计算吞吐 |
| kernel time | μs | GPU kernel 实际执行时间 |

## Benchmark 模板

```python
import torch
import ninetoothed
from time import perf_counter

def benchmark_kernel(kernel_fn, torch_fn, *args, warmup=10, repeats=100):
    # 预热
    for _ in range(warmup):
        kernel_fn(*args)
        torch_fn(*args)
    torch.cuda.synchronize()

    # 测量 ninetoothed
    start = perf_counter()
    for _ in range(repeats):
        kernel_fn(*args)
    torch.cuda.synchronize()
    kernel_time = (perf_counter() - start) / repeats

    # 测量 torch baseline
    start = perf_counter()
    for _ in range(repeats):
        torch_fn(*args)
    torch.cuda.synchronize()
    torch_time = (perf_counter() - start) / repeats

    return {"kernel_ms": kernel_time * 1000, "torch_ms": torch_time * 1000}
```

## 报告格式

所有 benchmark 结果记录为 Markdown 表格，包含：

| Operator | Shape | Dtype | BlockSize | Kernel(ms) | Torch(ms) | Speedup |
|----------|-------|-------|-----------|------------|-----------|---------|
| add | (1024,) | fp32 | 1024 | 0.012 | 0.015 | 1.25x |
