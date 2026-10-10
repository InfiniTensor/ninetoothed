# Benchmark — example 01

## When to run

Only after correctness PASS (SKILL.md D8).

## Commands

```bash
python skills/competition/ninetoothed-op-dev-skill/examples/01_elementwise_broadcast_add/verify.py \
  --benchmark \
  --json-output logs/benchmark/final-runtime-02/broadcast_add.json \
  --markdown-output logs/benchmark/final-runtime-02/broadcast_add.md
```

## Protocol (Phase 2B)

| Item | Value |
|------|-------|
| Shapes | `(128,256)`, `(512,512)`, `(2048,2048)` |
| dtype / device | float32 / CUDA |
| Output buffers | preallocated `out=` |
| Timer | `torch.cuda.Event(enable_timing=True)` |
| Warmup / repeats / inner | **30 / 30 / 100** |
| Stats | median, p10, p90; `spread_ratio=p90/median` |

## Recorded results (`logs/benchmark/final-runtime-02/broadcast_add.md`)

- device: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- ninetoothed_commit: `ef4c52899f5f836e3d77001b56c3544849cacf2c`
- measurement_scope: preallocated-output steady-state CUDA Event

| M | N | NT median | Torch median | ratio | NT stab | Torch stab |
|---|---|----------:|-------------:|------:|---------|------------|
| 128 | 256 | 0.125396 | 0.121555 | 1.0316 | stable | stable |
| 512 | 512 | 0.120316 | 0.115456 | 1.0421 | stable | stable |
| 2048 | 2048 | 0.049763 | 0.047555 | 1.0464 | noisy | stable |

Size inversion warnings (disclosed): NT/Torch (512,512)→(2048,2048) median decrease.
