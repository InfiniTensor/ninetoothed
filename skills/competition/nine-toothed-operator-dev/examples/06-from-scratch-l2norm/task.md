# Self-Test 06: From-Scratch Row-Wise L2 Normalization (Reduction/Block)

## Category

Reduction / block operator, written from scratch (no existing implementation in the repository).

## Input Task Statement

Implement a NineToothed row-wise L2 normalization operator: `output[i, :] = input[i, :] / sqrt(sum(input[i, :]**2) + eps)` for 2-D fp16/fp32 contiguous tensors, matching `torch.nn.functional.normalize(input, p=2, dim=-1)` semantics. Include a non-power-of-two row length, correctness tests against PyTorch, and a benchmark.

## Agent Execution Summary

Following the skill workflow:

1. **Contract restated**: rank-2 input/output, reduce over last dim, one program per row; fp16 accumulates in fp32; eps as scalar parameter; tail elements beyond the row must not contribute to the sum.
2. **Repository recon**: nearest pattern is `rms_norm.py` — `tile((1, BLOCK_SIZE))` row arrangement, scalar `Tensor(0)` for eps, `ntl.cast(input, ntl.float32)` accumulation, `BLOCK_SIZE=input.shape[-1]` at launch. `softmax` (core tests) shows `other=` fill for tail tiles.
3. **Arrangement**: `input.tile((1, BLOCK_SIZE))`, `eps` passed through, `output.tile((1, BLOCK_SIZE))`.
4. **Key design decision**: `Tensor(2, other=0)` — 0 is the identity for sum-of-squares, so tail elements of the padded block (row length 781 is not a power of two; Triton pads to `next_power_of_2`) contribute nothing. Choosing the wrong fill (e.g. `-inf` as in softmax) would poison the sum.
5. **Tests**: odd sizes (1823, 781), fp16/fp32, boundary (1,1), large square (4096, 4096).
6. **Benchmark**: `triton.testing.do_bench` vs `torch.nn.functional.normalize`, three shapes including non-power-of-two reduction length.

## Correctness

Command:

```shell
python gpu-session/selftest_l2norm.py
```

Result (RTX 4090, torch 2.9.1+cu128, Triton 3.5.1, NineToothed 0.25.0):

```text
correctness shape=(1823, 781) dtype=torch.float32 allclose=True max_abs_diff=2.980e-08
correctness shape=(1823, 781) dtype=torch.float16 allclose=True max_abs_diff=6.104e-05
correctness shape=(1, 1)      dtype=torch.float32 allclose=True max_abs_diff=0.000e+00
correctness shape=(4096, 4096) dtype=torch.float32 allclose=True max_abs_diff=1.490e-08
ALL CORRECTNESS PASSED
```

## Benchmark

Baseline: `torch.nn.functional.normalize(input, p=2, dim=-1)`. Timing: `triton.testing.do_bench`, fp16, contiguous.

```text
benchmark shape=(4096, 781)   ninetoothed=0.0215ms torch=0.0270ms ratio=0.80x
benchmark shape=(4096, 4096)  ninetoothed=0.0761ms torch=0.1030ms ratio=0.74x
benchmark shape=(16384, 1024) ninetoothed=0.0760ms torch=0.1059ms ratio=0.72x
```

Conclusion: the fused single-kernel NineToothed implementation is 20-28% faster than PyTorch eager on all tested shapes, because `F.normalize` launches separate norm/divide kernels while this implementation reads the row once, reduces in registers, and writes once. This is a real, evidenced performance win (not just parity).

## Unsupported Cases

- `ndim != 2` inputs (guarded by assertion).
- Rows longer than the maximum Triton block size for this arrangement (single-block-per-row design; very long rows would need a multi-block two-pass reduction).
- Non-contiguous inputs are untested for this operator.
- p != 2 norms and reduction over non-last dims are out of scope.
