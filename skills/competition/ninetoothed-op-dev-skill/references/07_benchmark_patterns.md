# 07 — Benchmark patterns

## When to use

Performance-sensitive hidden tasks, regression analysis, competition materials (≥2 self-test tasks need benchmark).

## Optional examples (`--examples-root`)

From `<examples-root>/README.md`:

```bash
cd <examples-root>
pytest -m benchmark
pytest -m benchmark -k TestMM
```

⚠️ **User environment action:** `pip install -e .` in the examples repo before pytest there.

## Autotuning warning

`Symbol(..., meta=True)` can make benchmarks **very slow**. For controlled benchmarks:

- Set `constexpr=True` and pass block size at launch, or  
- Replace `Symbol` with fixed int (see examples README)

Also disable matching Triton autotuning when comparing fairly.

## What to record (competition rubric)

| Field | Example |
|-------|---------|
| Baseline | PyTorch `torch.add` / same-kernel fair baseline |
| Input sizes | e.g. (128,256), (512,512), (2048,2048) |
| Device | RTX 5070 Ti Laptop, CUDA |
| Timer | `torch.cuda.Event(enable_timing=True)` |
| Stats | **median / p10 / p90**; `spread_ratio=p90/median`; stability labels |
| Fair baseline | same dtype/device/size; disclose `meta=True` / autotuning |
| Conclusion bounds | what is comparable vs incomparable; never invent numbers |
| Command | exact verify/`run_benchmark.py` invocation |
| Raw output | `<repo-root>/logs/benchmark/*.md` + `.json` |

## bench.py helpers

`<examples-root>/bench.py`:

- `assert_match(impls, args)` — correctness across providers before timing  
- Plotting utilities for manual analysis  

## Compliance

- No fabricated timings  
- Same input shapes for baseline and candidate  
- Note if autotuning was disabled (affects absolute numbers)

## Warmup (required for fair timing)

Before recording benchmark numbers:

1. Run the kernel **≥5 times** with the same shapes (JIT + cache warmup).
2. When sweeping `BLOCK_SIZE`, warmup **each** block size separately.
3. Document `WARMUP_ITERS` in test or script when embedding micro-benchmarks in pytest.
