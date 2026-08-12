# Performance Benchmark Guide

## Gate Timing With Correctness

Run the candidate and reference on the exact timed inputs and require the
configured comparison to pass. Do not publish timing when the correctness gate
fails or when the candidate silently falls back to another path.

## Record The Experiment

Capture:

- Repository revision and dirty status.
- Operator, shape, dtype, layout, and device.
- Baseline and candidate call paths.
- Warmup count, repeat count, synchronization, and timer type.
- Compile or first-call time separately from steady-state samples.
- Raw samples or a compact machine-readable result.
- Mean, median, minimum, variability, and derived ratio.

## Controlled Short Run

Use the bundled script after installing the repository environment:

```bash
python scripts/run_ntops_microbenchmark.py \
  --task-id local-check \
  --operator add \
  --shape 1024x1024 \
  --dtype float32 \
  --warmup 10 \
  --repeat 30 \
  --output-csv benchmark.csv \
  --output-md benchmark.md
```

The script requires CUDA and leaves timing unset when its correctness guard
cannot pass.

## Interpret Conservatively

- Compare on the same device and process where practical.
- Treat very small kernels near timer resolution cautiously.
- Rerun noisy results before diagnosing a regression.
- Inspect generated source and launch configuration when a regression is real.
- Report the exact selected shape instead of claiming broad speedup.
- Keep long benchmark suites outside a focused debugging loop unless requested.

## Regression Closure

Reproduce, profile or inspect the likely bottleneck, make one minimal change,
rerun correctness, and repeat the same timing protocol. Preserve an unfavorable
baseline result when it is real.
