# Performance and Benchmarking

Use this reference for benchmark tasks, performance-sensitive operators, generated source review, and regression analysis.

## Benchmark Contract

Every benchmark note must include:

- operator and implementation names
- input sizes and parameter sweep
- dtype and device
- baseline implementation
- command
- observed timing or benchmark table path
- conclusion: faster, comparable, slower, or blocked

Do not report "optimized" without data.

## Candidate Commands

The core `ninetoothed` repository has NO built-in benchmark suite; it only uses `triton.testing.do_bench` internally (auto_tuner, build). For core-repository performance tasks, write a small script with `triton.testing.do_bench`:

```python
import triton.testing

ms_nt = triton.testing.do_bench(lambda: ninetoothed_op(*args))
ms_torch = triton.testing.do_bench(lambda: torch_reference(*args))
```

The `ninetoothed-examples` repository does have benchmark infrastructure. Adapt to whichever repository the task targets:

```shell
pytest -m benchmark -k TestAddBenchmark -q
pytest -m benchmark -k TestSoftmaxBenchmark -q
python run_experiments.py
python evaluate_performance.py
```

For a narrow correctness-before-performance check in `ninetoothed-examples`:

```shell
pytest tests/test_ops.py -k add -q
pytest tests/test_benchmarks.py -k TestAddBenchmark -q -m benchmark
```

## Benchmark Design

Elementwise:

- Sweep total elements across powers of two and at least one odd size.
- Compare NineToothed with PyTorch and Triton if present.
- Watch memory bandwidth and redundant load/store.

Reduction:

- Sweep reduction length and batch rows.
- Include non-power-of-two reduction sizes.
- Check numerical tolerance and stable formula.

Layout-sensitive:

- Compare contiguous and non-contiguous views.
- Report whether a copy is made.
- State whether the slowdown is expected due to stride access.

## Optimization Heuristics

Prefer simple, justified changes:

- reduce redundant loads and stores
- avoid recomputing broadcasted expressions inside loops
- use stable reduction formulas
- choose tile/block sizes aligned with existing operators
- preserve contiguous fast paths when adding layout support
- avoid converting every input to contiguous unless the task permits it
- keep benchmark tolerances separate from correctness tolerances

## Regression Diagnosis

When slower than baseline:

1. Verify correctness first.
2. Compare input sizes and dtype with baseline.
3. Inspect whether layout created strided memory access.
4. Inspect generated source if the repository exposes it.
5. Check AOT/build flags and cache behavior.
6. Propose the smallest fix or a documented limitation.

Regression note template:

```text
Symptom:
Baseline:
Candidate:
Input scale:
Command:
Observed result:
Likely cause:
Minimal fix or mitigation:
Verification:
```
