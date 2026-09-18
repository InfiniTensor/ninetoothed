# Task card — block-size performance regression

| Field | Value |
|-------|-------|
| Math | 1-D elementwise `out = a + b` |
| Shape | fixed `N` (document in bench output; default 98432) |
| dtype | float32 |
| Broadcast | N/A |
| Layout | contiguous |
| Boundaries | correctness PASS at each `BLOCK_SIZE` before timing |
| Reference | `tests/test_generation.py`; `references/07_benchmark_patterns.md` |
| Tests | `tests/test_example_block_size.py` + `verify.py` |
| Benchmark | D8: warmup ≥5, repeated timing, compare BLOCK 32 vs 1024 |
| Unsupported | peak-perf claims under undisclosed `meta=True` autotuning |
