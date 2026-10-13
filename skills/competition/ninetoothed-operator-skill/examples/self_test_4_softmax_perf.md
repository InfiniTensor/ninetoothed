# Self-test Task 4: Performance Analysis and Benchmark

## 1. Task description

- **Operator type:** Softmax (revisited) — performance diagnosis and benchmark
- **Inputs:** 2D float32 tensor across multiple shapes
- **Focus:** NineToothed softmax throughput vs PyTorch baseline,
  generated-source dump trigger, BLOCK_SIZE padding waste analysis
- **Hardware:** Google Colab T4 GPU
- **NineToothed version:** 0.26.0

## 2. Agent execution summary

**Files inspected:**
- `ops/softmax.py` — kernel implementation with Symbol-based BLOCK_SIZE
- `tests/test_softmax_perf.py` — benchmark + generated-source dump + padding waste tests
- NineToothed `NINETOOTHED_DUMP_GENERATED_SOURCE` env var for IR inspection

**Issue discovered during task:**
- `test_softmax.py` used `ninetoothed.block_size()` autotuning (no BLOCK_SIZE kwarg needed)
- `test_softmax_perf.py` required explicit `BLOCK_SIZE=` kwarg for controlled benchmarking
- These two calling conventions conflicted after switching to `Symbol("BLOCK_SIZE", constexpr=True)`

**Fix applied:**
- Introduced `_SoftmaxKernel` wrapper class in `ops/softmax.py`
- Without BLOCK_SIZE kwarg: auto-infers `triton.next_power_of_2(input.shape[-1])`
- With BLOCK_SIZE kwarg: passes through directly
- Both test files now work without modification

**Files modified:**
- `ops/softmax.py` — added `_SoftmaxKernel` wrapper, preserved all existing logic

## 3. Correctness test

```bash
pytest tests/test_softmax.py tests/test_softmax_perf.py -v
```

Result:
```
tests/test_softmax.py::test_softmax_correctness[shape0] PASSED
tests/test_softmax.py::test_softmax_correctness[shape1] PASSED
tests/test_softmax.py::test_softmax_correctness[shape2] PASSED
tests/test_softmax.py::test_softmax_correctness[shape3] PASSED
tests/test_softmax.py::test_softmax_correctness[shape4] PASSED
tests/test_softmax.py::test_softmax_row_sums_to_one PASSED
tests/test_softmax.py::test_softmax_numerical_stability_large_logits PASSED
tests/test_softmax.py::test_softmax_numerical_stability_negative_logits PASSED
tests/test_softmax.py::test_softmax_noncontiguous_rows_fallback PASSED
... 10/10 passed (test_softmax.py) + 10/10 passed (test_softmax_perf.py)
```

## 4. Benchmark results

### 4.1 NineToothed vs PyTorch throughput (test_softmax.py)

Command:
```bash
pytest tests/test_softmax.py -v -s -k benchmark
```

| Shape | NineToothed | PyTorch | Winner |
|-------|-------------|---------|--------|
| (1024, 512)  | 136.8 GB/s (0.031ms) | 13.8 GB/s (0.303ms) | NT in this run |
| (1024, 2048) | 225.6 GB/s (0.074ms) | 208.0 GB/s (0.081ms) | NT in this run |
| (4096, 4096) | 227.8 GB/s (0.589ms) | 242.3 GB/s (0.554ms) | PyTorch in this run |

### 4.2 NineToothed throughput across shapes (test_softmax_perf.py)

Command:
```bash
pytest tests/test_softmax_perf.py -v -s -k throughput
```

| Shape | NineToothed | PyTorch |
|-------|-------------|---------|
| (512, 256)   | 17.5 GB/s  | 42.3 GB/s |
| (512, 1024)  | 150.6 GB/s | 205.1 GB/s |
| (512, 4096)  | 227.8 GB/s | 134.2 GB/s |
| (2048, 4096) | 235.5 GB/s | 145.5 GB/s |

### 4.3 BLOCK_SIZE padding waste analysis

```bash
pytest tests/test_softmax_perf.py -v -s -k block_size
```

Result:
```
ncols=100: BLOCK_SIZE=128: 0.028ms | BLOCK_SIZE=1024: 0.026ms | speedup: 0.91x
```

For ncols=100 on T4, BLOCK_SIZE=128 (optimal) and BLOCK_SIZE=1024 (10× over-padded) show
a small timing difference in this run. The rule "use smallest power-of-2 ≥ ncols"
is still the documented default, while exact microbenchmark ratios remain
environment-dependent.

## 5. Performance analysis

**NineToothed advantage observed in this run:** wide rows in the separated
throughput benchmark.
- At (512, 4096): NT 227.8 GB/s vs PT 134.2 GB/s
- At (2048, 4096): NT 235.5 GB/s vs PT 145.5 GB/s

**PyTorch advantage range:** small ncols (≤ 512).
- At (512, 256): NT 17.5 GB/s vs PT 42.3 GB/s

**Conclusion:** this benchmark records both favorable and unfavorable shapes.
Use the raw log in `evidence/pytest-benchmark.txt` for exact numbers rather
than treating a single Colab run as a fixed performance guarantee.

## 6. Generated-source dump trigger

`NINETOOTHED_DUMP_GENERATED_SOURCE=1` was verified in `test_generated_source_dump_trigger`.
The test passed, confirming the dump trigger and post-dump correctness for a
(8, 256) input with `BLOCK_SIZE=256`.
