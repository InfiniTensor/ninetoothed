# Self-Test 01: Elementwise Add With Broadcast Awareness

## Category

Elementwise / broadcast operator.

## Input Task Statement

Implement or repair a NineToothed elementwise add-style operator. The operator should accept two tensors, produce an output matching PyTorch addition semantics for the selected supported shapes, and include correctness tests.

Required coverage:

- fp32 correctness
- at least one non-power-of-two size
- explicit note on whether broadcasting is supported
- no unrelated refactor

## Expected Agent Workflow

1. Search for existing add, silu, swiglu, or elementwise kernels.
2. Restate shape, dtype, broadcast, and layout assumptions.
3. Implement the smallest operator or patch.
4. Compare against `torch.add`.
5. Run targeted pytest.
6. Run benchmark if the final environment has CUDA.

## Produced Patch Summary

No source patch was needed for this public-repository self-test. The task verified that the existing NineToothed add operator and the skill's elementwise guidance match the repository pattern.

## Correctness

Command:

```shell
/usr/local/miniconda3/envs/py312/bin/python -m pytest tests/test_ops.py::TestAdd::test_correctness -q
```

Result:

```text
1 passed in 5.58s
```

## Benchmark

Command:

```shell
/usr/local/miniconda3/envs/py312/bin/python -m pytest tests/test_benchmarks.py::TestAddBenchmark::test_benchmark -q -m benchmark
```

Result:

```text
1 passed in 9.80s
```

## Performance Conclusion

Additional custom timing on RTX 4090, torch 2.9.1+cu128, fp16 shape `(98432,)`:

```text
add allclose True
add ninetoothed_ms 0.05119999870657921
add torch_ms 0.020479999482631683
```

Conclusion: correctness is exact for this fp16 elementwise case. PyTorch is faster on the small custom timing, so the skill should require benchmark evidence before claiming an optimization.

## Unsupported Cases

This self-test does not prove broadcasting or non-contiguous support. A separate layout task shows non-contiguous add mismatch in the public examples wrapper.
