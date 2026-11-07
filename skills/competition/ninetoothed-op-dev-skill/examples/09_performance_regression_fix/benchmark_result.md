# Benchmark — example 09

## When to run

Only after correctness PASS at **each** `BLOCK_SIZE` (SKILL.md D8).

## Commands

```bash
python skills/competition/ninetoothed-op-dev-skill/examples/09_performance_regression_fix/verify.py \
  --benchmark \
  --json-output logs/benchmark/final-runtime-02/block_size.json \
  --markdown-output logs/benchmark/final-runtime-02/block_size.md
```

## Protocol (Phase 2B)

| Item | Value |
|------|-------|
| Op | 1-D add, constexpr `BLOCK_SIZE` (no `meta=True`) |
| Sizes | `N ∈ {98432, 1048576, 4194304}` |
| Configs | `BLOCK_SIZE ∈ {32, 256, 1024}` |
| Timer | `torch.cuda.Event(enable_timing=True)` |
| Warmup / repeats / inner | **30 / 30 / 100** |
| Stats | median/p10/p90; spread_ratio; stability; fastest **per size** |

## Recorded results (`logs/benchmark/final-runtime-02/block_size.md`)

- device: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- ninetoothed_commit: `ef4c52899f5f836e3d77001b56c3544849cacf2c`
- size_dependence: yes

| N | BLOCK_SIZE | median ms | p10 | p90 | spread | stability |
|---|------------|----------:|----:|----:|-------:|-----------|
| 98432 | 32 | 0.125983 | 0.049237 | 0.157446 | 1.250 | stable |
| 98432 | 256 | 0.130278 | 0.078267 | 0.147752 | 1.134 | stable |
| 98432 | 1024 | 0.128404 | 0.075815 | 0.150004 | 1.168 | stable |
| 1048576 | 32 | 0.087471 | 0.055216 | 0.107493 | 1.229 | stable |
| 1048576 | 256 | 0.072024 | 0.058294 | 0.114206 | 1.586 | stable |
| 1048576 | 1024 | 0.071493 | 0.032741 | 0.104581 | 1.463 | stable |
| 4194304 | 32 | 0.557959 | 0.528783 | 0.594734 | 1.066 | stable |
| 4194304 | 256 | 0.133946 | 0.121333 | 0.164150 | 1.225 | stable |
| 4194304 | 1024 | 0.134427 | 0.114178 | 0.156808 | 1.166 | stable |

Fastest by size (median): N=98432→32; N=1048576→1024; N=4194304→256.  
At N=4,194,304, BS=32 median is ~4.16× slower than BS=256/1024.
