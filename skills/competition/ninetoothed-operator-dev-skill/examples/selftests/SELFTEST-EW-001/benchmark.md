# ntops Microbenchmark

## Environment
- timestamp: `2026-07-12T12:45:49+00:00`
- task_id: `SELFTEST-EW-001`
- operator: `add`
- shape: `1024x1024`
- dtype: `float32`
- device: `NVIDIA GeForce RTX 4090`
- torch_version: `2.6.0a0+ecf3bae40a.nv25.01`
- torch_cuda: `12.8`

## Command

```text
run_ntops_microbenchmark.py --task-id SELFTEST-EW-001 --operator add --shape 1024x1024 --dtype float32 --warmup 10 --repeat 30 --baseline pytorch --candidate ntops --output-csv outputs/ew_benchmark.csv --output-md outputs/ew_benchmark.md
```

## Correctness
- correctness_status: `PASS`

## Timing Summary

| Metric | Baseline | Candidate |
| --- | --- | --- |
| mean ms | `0.017958` | `0.067030` |
| median ms | `0.017408` | `0.066000` |
| min ms | `0.016448` | `0.064512` |

- baseline_samples_ms: `0.024576;0.020480;0.018432;0.018432;0.018560;0.017408;0.017408;0.018432;0.017408;0.017408;0.017408;0.017408;0.017408;0.016480;0.018432;0.017408;0.017472;0.018432;0.017408;0.017408;0.018336;0.017312;0.016448;0.017408;0.017408;0.018400;0.017408;0.017408;0.017408;0.017408`
- candidate_samples_ms: `0.069632;0.067584;0.065536;0.067584;0.066560;0.067584;0.065536;0.066528;0.089088;0.070592;0.066560;0.065536;0.065536;0.065536;0.065536;0.066560;0.066560;0.066464;0.065536;0.066560;0.065536;0.064512;0.064512;0.065536;0.066560;0.064512;0.065536;0.065536;0.065536;0.066528`

- speedup_median: `0.263758`

## Caveats
- single GPU
- single-server short benchmark
- not official hidden benchmark
- no long benchmark
- no AOT conclusion
- no fake timing; timing fields stay NA when CUDA or correctness preflight fails
- Interpret this as a controlled short benchmark, not a broad performance conclusion.
