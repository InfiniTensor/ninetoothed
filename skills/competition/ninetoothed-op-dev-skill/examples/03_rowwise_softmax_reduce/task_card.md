# Task card — row-wise softmax

| Field | Value |
|-------|-------|
| Math | row-wise softmax (stable: max → exp → sum) |
| Shape | input `(M,N)` → output `(M,N)` |
| dtype | float32 |
| Broadcast | N/A |
| Layout | contiguous inputs/outputs |
| Boundaries | M,N ≥ 1; numerical stability (max before exp); CUDA preferred |
| Reference | `tests/test_softmax.py`; `torch.softmax(..., dim=-1)` |
| Tests | upstream `pytest tests/test_softmax.py` and/or `verify.py` |
| Benchmark | N/A for default demo (correctness-only) |
| Unsupported | claiming peak perf without D8; rewriting compiler core |
