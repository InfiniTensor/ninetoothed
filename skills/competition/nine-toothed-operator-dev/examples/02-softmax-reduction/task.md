# Self-Test 02: Softmax Reduction

## Category

Reduction / block operator.

## Input Task Statement

Implement or repair a row-wise softmax-style NineToothed operator over the last dimension. The implementation must use a numerically stable formula and compare against `torch.softmax(input, dim=-1)`.

Required coverage:

- non-power-of-two reduction length
- stable subtract-max formulation
- identity fill for masked tail values if needed
- correctness tolerance justified by dtype
- benchmark over multiple reduction sizes

## Expected Agent Workflow

1. Search for existing softmax, rms_norm, and reduction examples.
2. Restate axis, shape, dtype, and boundary behavior.
3. Use `ntl.max`, `ntl.exp`, and `ntl.sum` in the application.
4. Compare against PyTorch.
5. Run targeted pytest.
6. Run softmax benchmark if available.

## Produced Patch Summary

No source patch was needed for the core repository softmax smoke test. The public examples three-way softmax test exposed a strict Triton exact-match comparison failure, which is recorded as diagnostic evidence.

## Correctness

Command:

```shell
/usr/local/miniconda3/envs/py312/bin/python -m pytest tests/test_softmax.py -q
```

Result:

```text
1 passed in 1.86s
```

Public examples three-way comparison:

```shell
/usr/local/miniconda3/envs/py312/bin/python -m pytest tests/test_ops.py::TestSoftmax::test_correctness -q
```

Result:

```text
1 failed in 5.67s
AssertionError: NineToothed and Triton outputs differ.
```

## Benchmark

Command:

```shell
/usr/local/miniconda3/envs/py312/bin/python -m pytest tests/test_benchmarks.py::TestSoftmaxBenchmark::test_benchmark -q -m benchmark
```

Result:

```text
1 failed in 7.85s
Failure occurred during benchmark correctness precheck because Triton exact-match tolerance was set to {"atol": 0, "rtol": 0}.
```

## Performance Conclusion

Additional custom timing on RTX 4090, torch 2.9.1+cu128, fp16 shape `(4096, 781)`:

```text
softmax allclose_atol_1e-3 True
softmax max_abs_diff_vs_torch 1.52587890625e-05
softmax ninetoothed_ms 0.05432000011205673
softmax torch_ms 0.021503999829292297
```

Conclusion: NineToothed softmax matches PyTorch within fp16 tolerance on the tested shape. The public examples benchmark is blocked by an exact-match Triton comparison, not by a PyTorch mismatch.

## Unsupported Cases

This self-test covers fp16 and fp32-style repository smoke tests only. It does not prove all dynamic shapes, all reduction axes, or non-contiguous layouts.
