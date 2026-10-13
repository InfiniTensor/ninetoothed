# Benchmark Guide

## Minimum Requirements

Every benchmark must state:
1. **Baseline**: PyTorch reference (or upstream implementation)
2. **Input size**: concrete shape, dtype, device
3. **Command**: full reproducible command
4. **Result**: ms/iter, ratio (nt/torch)
5. **Conclusion**: whether performance is acceptable and why

## Implementation Pattern

```python
def _benchmark(fn, warmup=10, repeats=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats
```

Key points:
- Always `torch.cuda.synchronize()` before and after timing.
- Warmup excludes JIT compilation overhead.
- Report ms/iter and ratio vs PyTorch baseline.
- If NineToothed is slower, record hypothesis (block size, memory access, etc.).

## Generated Source & AOT Build

For performance-sensitive tasks:
1. After first kernel call, inspect generated Triton/C++ source in temp directory.
2. Verify load/store count, reduction dimension, mask/boundary coverage.
3. For AOT build (`ninetoothed.build`): check `output_dir` for `.so` and CSV files.
4. Compare ms/iter with and without AOT to verify cache hits.
