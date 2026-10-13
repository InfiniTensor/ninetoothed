# Self-Test 05: From-Scratch GELU Operator (Elementwise)

## Category

Elementwise operator, written from scratch (no existing implementation in the repository).

## Input Task Statement

Implement a NineToothed GELU operator (tanh approximation, matching `torch.nn.functional.gelu(x, approximate="tanh")`) for 1-D fp16/fp32 contiguous tensors. Include correctness tests against PyTorch covering a non-power-of-two size and a boundary size, plus a benchmark against PyTorch.

## Agent Execution Summary

Following the skill workflow:

1. **Contract restated**: input/output rank-1, same shape and dtype; fp16 and fp32; tanh-approximation formula `0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))`; contiguous-only (documented); rank must match kernel declaration (lesson from self-test 03).
2. **Repository recon**: nearest patterns are `silu.py` (elementwise, `ntl.cast(x, ntl.float32)` for fp16 stability, `ntl.sigmoid`) and `add.py` (multi-input tile). `tanh` is not in `ntl`; found via `ninetoothed.language.libdevice` (`from triton.language.extra import libdevice`), consistent with `libdevice.sin/pow` usage in core tests.
3. **Arrangement**: `input.tile((BLOCK_SIZE,))`, `output.tile((BLOCK_SIZE,))`, `BLOCK_SIZE=1024` at launch, `Symbol("BLOCK_SIZE", constexpr=True)` — same as silu.
4. **Implementation**: arrange-and-apply module style, wrapper asserts `ndim == 1`.
5. **Tests**: fp32/fp16 at 98432 (repo-idiomatic size), 781 (non-power-of-two), 1 (boundary).
6. **Benchmark**: `triton.testing.do_bench` vs PyTorch, sweep 2^16 / 2^20 / 2^24 and a non-power-of-two 10M.

## Failure Diagnosis (closed loop)

- **Symptom**: first run failed at Triton compile time: `NameError('SQRT_2_OVER_PI is not defined')`.
- **Root cause**: the application function's source is extracted and compiled standalone by the NineToothed code generator; module-level Python globals are not captured into the generated Triton kernel.
- **Minimal fix**: inline numeric constants into the application body.
- **Re-run**: all correctness checks passed (below). Lesson recorded: *never reference module-level constants inside `application`; inline them or pass as `Tensor(0)` / constexpr symbols.*

## Correctness

Command:

```shell
python gpu-session/selftest_gelu.py
```

Result (RTX 4090, torch 2.9.1+cu128, Triton 3.5.1, NineToothed 0.25.0):

```text
correctness shape=(98432,) dtype=torch.float32 allclose=True max_abs_diff=2.384e-07
correctness shape=(98432,) dtype=torch.float16 allclose=True max_abs_diff=0.000e+00
correctness shape=(781,) dtype=torch.float32 allclose=True max_abs_diff=2.384e-07
correctness shape=(1,) dtype=torch.float32 allclose=True max_abs_diff=0.000e+00
ALL CORRECTNESS PASSED
```

## Benchmark

Baseline: `torch.nn.functional.gelu(input, approximate="tanh")`. Timing: `triton.testing.do_bench`, fp16, contiguous.

```text
benchmark size=65536    ninetoothed=0.0046ms torch=0.0044ms ratio=1.05x
benchmark size=1048576  ninetoothed=0.0083ms torch=0.0080ms ratio=1.04x
benchmark size=16777216 ninetoothed=0.0766ms torch=0.0777ms ratio=0.99x
benchmark size=10000000 ninetoothed=0.0479ms torch=0.0480ms ratio=1.00x
```

Conclusion: no meaningful regression versus PyTorch across all sizes (0.99x–1.05x). The operator is memory-bound; at large sizes NineToothed matches or slightly beats PyTorch eager. No optimization claim is made for small sizes, where the ~5% gap is launch overhead.

## Unsupported Cases

- Inputs with `ndim != 1` (guarded by assertion; see self-test 03 for the silent-failure mode this prevents).
- Non-contiguous inputs (wrapper does not flatten-copy; documented limitation).
- dtypes other than fp16/fp32 are untested.
