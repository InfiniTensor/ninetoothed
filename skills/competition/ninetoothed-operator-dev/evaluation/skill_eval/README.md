# skill_eval — KernelSwift-inspired evaluation utilities

**English** | [中文](README.zh-CN.md)

Method reference: KernelSwift Agent Infra (Shanghai AI Lab), public technical
talk. Independent implementation.

## What we adapt here

| Concept | This module | Implementation |
|---|---|---|
| Fixed computation graph | `robust_benchmark()` warmup phase | kernel_fn called `warmup=25` times before timing starts |
| Repeated measurement | `robust_benchmark(iters=100)` | CUDA Events per-iteration |
| Outlier removal | `_remove_outliers_iqr()` | Tukey IQR fence, k=1.5 |
| Reward-hacking guard (static) | `static_analysis(source_path)` | AST: count tl.load/tl.store, detect zero-stores |
| Reward-hacking guard (dynamic) | `dynamic_analysis(kernel_fn, output)` | all-zero / NaN / Inf / no-op checks |
| Reward-hacking guard (ncu roofline) | `ncu_roofline_check()` | stub (ncu must be installed) |
| Measurement caching | (TODO: add SQLite or JSON cache) | not yet implemented |
| Island-based evolution | (out of scope for v0) | Stage 3 experimental |

## Usage

Replace `bench_compare.benchmark(...)` with `robust_benchmark(...)` to get
outlier removal and reward-hacking detection in one call:

```python
from skill_eval import robust_benchmark, full_guard, BenchResult

# 1. Run the kernel once to get the output tensor.
out = torch.empty_like(x)
kernel(x, out, BLOCK_SIZE=512)

# 2. Run robust benchmark.
result: BenchResult = robust_benchmark(
    kernel_fn=lambda: kernel(x, out, BLOCK_SIZE=512),
    bytes_moved=x.numel() * x.element_size() * 2,
    # gpu auto-detected from the live device; pass gpu= or set $NT_GPU to pin it.
)
print(result)

# 3. Run full guard (static + dynamic).
report = full_guard(
    kernel_fn=lambda: kernel(x, out, BLOCK_SIZE=512),
    output_tensor=out,
    input_tensor=x,
)
print(report)  # "reward_hacking: CLEAN" or "SUSPICIOUS: ..."
```

## Why outlier removal matters

GPU benchmark noise sources:
- Thermal throttling (first N iters may be cold)
- OS jitter (random L3 eviction, scheduler preemption)
- CUDA driver synchronisation overhead (first call)

IQR fence at k=1.5 removes ~7% of samples on a stable GPU, but catches the
occasional 10x spike that would inflate the mean by 15-20%. On a noisy server
the improvement is larger.

## Differences from raw bench_compare.py

| Metric | bench_compare.py | robust_benchmark() |
|---|---|---|
| Outlier removal | None | IQR fence |
| Reward hacking | None | Bandwidth + AST + dynamic |
| Fixed comp graph | Implicit | Explicit warmup phase |
| Output | dict | BenchResult dataclass |
