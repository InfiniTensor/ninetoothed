# Task card — broadcast add

| Field | Value |
|-------|-------|
| Math | `out = a + b` with NumPy/PyTorch broadcast |
| Shape | a `(M,1)`, b `(1,N)`, out `(M,N)` |
| Correctness dtype | float32 and float16 (fp16: explicit atol/rtol) |
| Benchmark dtype | float32 only |
| Broadcast | yes — column vector + row vector |
| Layout | contiguous inputs; out contiguous |
| Boundaries | M,N ≥ 1; CUDA preferred, CPU OK for smoke |
| Reference | `tests/test_add.py`, `tests/test_expand.py`; `torch.add` |
| Tests | `tests/test_example_broadcast_add.py` + `verify.py` |
| Benchmark | optional micro-bench vs `torch.add` (float32 shapes only) |
| Unsupported | int dtypes; autotuning claims; no float16 performance claims |
