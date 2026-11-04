# Benchmarking And Diagnostics

Use this reference when a task requires performance measurement, generated
source visibility, or AOT build triage.

## Benchmark Requirements

For performance-sensitive tasks, record:

- Input shapes and dtypes
- Baseline provider, usually PyTorch
- Timing method and warmup/repeat counts
- Throughput formula, such as GB/s
- Result table and conclusion
- Whether non-contiguous timing includes copy overhead

Do not treat `assert ms > 0` as a performance claim. It only proves the timer
ran. A useful benchmark compares providers or records a regression-relevant
metric.

## CUDA Event Timing Template

```python
for _ in range(3):
    fn()
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(50):
    fn()
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / 50
```

## Generated-Source Dump

The bundled self-test verifies the dump trigger and post-dump correctness.

```python
import os
os.environ["NINETOOTHED_DUMP_GENERATED_SOURCE"] = "1"
kernel(inputs...)
del os.environ["NINETOOTHED_DUMP_GENERATED_SOURCE"]
```

If a hidden task requires generated-source analysis, capture the dump output or
file, save it in the task evidence, and inspect relevant load/store/mask/
BLOCK_SIZE patterns.

## AOT Build Triage

AOT build is documented as a workflow item, not validated by this package's
self-tests. If a task asks for AOT:

1. Record the exact build command.
2. Record the target environment.
3. Save stdout/stderr.
4. Check unsupported dtype combinations and `Symbol(..., upper_bound=...)`.
5. Do not report AOT success unless the build actually ran.
