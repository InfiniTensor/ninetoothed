# ntops Microbenchmark

## Environment
- timestamp: `2026-07-12T12:45:54+00:00`
- task_id: `SELFTEST-RED-001`
- operator: `softmax`
- shape: `64x1024`
- dtype: `float32`
- device: `NVIDIA GeForce RTX 4090`
- torch_version: `2.6.0a0+ecf3bae40a.nv25.01`
- torch_cuda: `12.8`

## Command

```text
run_ntops_microbenchmark.py --task-id SELFTEST-RED-001 --operator softmax --shape 64x1024 --dtype float32 --warmup 10 --repeat 30 --baseline pytorch --candidate ntops --output-csv outputs/red_benchmark.csv --output-md outputs/red_benchmark.md
```

## Correctness
- correctness_status: `PASS`

## Timing Summary

| Metric | Baseline | Candidate |
| --- | --- | --- |
| mean ms | `0.024473` | `0.072396` |
| median ms | `0.023568` | `0.071728` |
| min ms | `0.022528` | `0.070656` |

- baseline_samples_ms: `0.033792;0.026624;0.025600;0.025600;0.024576;0.023584;0.023552;0.023648;0.022624;0.023552;0.023552;0.024512;0.023552;0.024576;0.022528;0.024576;0.023680;0.022560;0.023552;0.023552;0.032768;0.024576;0.023584;0.023552;0.022528;0.023552;0.024448;0.023424;0.022528;0.023424`
- candidate_samples_ms: `0.077824;0.074752;0.071680;0.072608;0.071680;0.070688;0.071744;0.072704;0.072704;0.071680;0.071712;0.071680;0.072704;0.070656;0.070656;0.073728;0.073696;0.073728;0.072704;0.070656;0.072704;0.072704;0.071680;0.071680;0.071680;0.071680;0.072608;0.074752;0.070720;0.071680`

- speedup_median: `0.328575`

## Caveats
- single GPU
- single-server short benchmark
- not official hidden benchmark
- no long benchmark
- no AOT conclusion
- no fake timing; timing fields stay NA when CUDA or correctness preflight fails
- Interpret this as a controlled short benchmark, not a broad performance conclusion.
