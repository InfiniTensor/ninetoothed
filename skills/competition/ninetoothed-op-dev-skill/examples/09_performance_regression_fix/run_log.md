# Run log — example 09

## D1 Task card

See `task_card.md`.

## D2–D3 Route + rg

Family: **perf / diagnosis**.

```bash
cd "$REPO"
rg -n "BLOCK_SIZE|constexpr|meta=True" tests/test_generation.py
rg -n "Symbol\\(\"BLOCK" tests/test_add.py tests/ -g "*.py" | head
```

Nearest patterns: constexpr `BLOCK_SIZE` tiling; avoid undisclosed `meta=True` when comparing configs.

## D4–D6 Minimal patch + correctness

- `solution/add_tunable.py` — 1-D add with constexpr `BLOCK_SIZE`
- `tests/test_example_block_size.py` — correctness at 32 and 1024

```bash
pytest .../examples/09_performance_regression_fix/tests -v --tb=short
```

## D8 Benchmark

```bash
python .../examples/09_performance_regression_fix/verify.py
```

Fixed `N`, warmup, timed iters, print ms/iter for both BLOCK sizes + slowdown ratio.
If ratio is absurd, see `failure_diagnosis.md`.
