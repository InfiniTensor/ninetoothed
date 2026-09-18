# Task card — non-contiguous transpose add

| Field | Value |
|-------|-------|
| Math | `out = a + b` |
| Shape | 2-D views, e.g. `(M,N)` matching after transpose view |
| dtype | float32 |
| Broadcast | N/A |
| Layout | **non-contiguous** inputs (transpose / empty_strided); out may be contiguous |
| Boundaries | `assert not a.is_contiguous()` (and b); no blind `.contiguous()` |
| Reference | `tests/test_clone.py`, `tests/test_data_ptr.py`; `torch.add` on same views |
| Tests | `tests/test_example_strided_add.py` + `verify.py` |
| Benchmark | N/A (layout correctness, not perf) |
| Unsupported | fixing layout by materializing with `.contiguous()` unless card allows |
