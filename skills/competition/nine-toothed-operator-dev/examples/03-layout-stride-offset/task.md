# Self-Test 03: Layout, Stride, and Offset

## Category

Layout-sensitive operator.

## Input Task Statement

Add or repair support for a non-contiguous input case in a NineToothed operator, or document and test a clear contiguous-only limitation when true support is outside scope.

Required coverage:

- a test that creates a non-contiguous tensor view
- comparison against PyTorch on the same view
- explicit handling of stride, offset, or documented limitation
- no unconditional `.contiguous()` unless the task permits a copy

## Candidate Test Inputs

```python
base = torch.randn((64, 128), dtype=dtype, device=device)
input = base[:, ::2]
assert not input.is_contiguous()
```

or:

```python
input = base.t()
```

## Expected Agent Workflow

1. Search for operators or wrappers that pass stride metadata.
2. Identify whether the target operator promises layout support.
3. Add the smallest layout-aware implementation or guard.
4. Add a non-contiguous correctness test.
5. Compare against PyTorch using the exact view.
6. Record performance impact if benchmarked.

## Produced Patch Summary

No source patch was applied. This self-test intentionally probed the existing public examples add wrapper with a non-contiguous view to expose a layout-sensitive failure mode that the skill must force the agent to detect and document.

## Correctness

Command:

```shell
/usr/local/miniconda3/envs/py312/bin/python /root/ninetoothed-skill-work/selftest_custom.py
```

Result:

```text
layout shape (256, 256) stride (512, 2) is_contiguous False
layout add allclose False
```

## Layout Evidence

```text
Input was created with `base[:, ::2]`, producing shape `(256, 256)`, stride `(512, 2)`, and `is_contiguous=False`. Existing public examples `ops.ninetoothed.torch.add` returned an output that did not exactly match PyTorch for this non-contiguous view.
```

## Root Cause Analysis

Traced through the NineToothed source (master `c9ebd49`):

1. The examples add kernel declares rank-1 parameters: `tensors = (Tensor(1), Tensor(1), Tensor(1))` (`ninetoothed-examples/ops/ninetoothed/kernels/add.py`). The generated launch function derives its parameter list from the declared rank (`src/ninetoothed/tensor.py:615-631`), so it only reads `input.size(0)` and `input.stride(0)` (`src/ninetoothed/torchifier.py:19-31`, `src/ninetoothed/generation.py:518-627`).
2. Passing a 2-D tensor to a rank-1 kernel does NOT raise. `tensor.size(0)`/`tensor.stride(0)` are valid calls on a 2-D tensor, so the launch silently treats the `(256, 256)` view as a 1-D tensor of length 256 with stride 512. The grid becomes `ceil(256 / 1024)` = 1 program.
3. Pointer arithmetic for the 1-D tile is `ptr + (pid * BLOCK_SIZE + arange(BLOCK_SIZE)) * stride_0` with mask `offsets < size_0` (`src/ninetoothed/generation.py:732-735`). With `stride_0=512`, `size_0=256`, the kernel reads exactly `a[i, 0]` for `i < 256` — only the first column of 65536 elements.
4. On the write side, `torch.empty_like(non_dense_view)` returns a contiguous tensor. The kernel writes only `output[i, 0]` (256 elements); the remaining 65280 elements keep uninitialized garbage. Hence `out[:, 0]` is correct and everything else is garbage: `allclose False` from both under-reading and under-writing.
5. There is no runtime ndim/contiguity validation on the JIT/torch path, and no non-contiguous test exists in the core `tests/` directory, so this is a silent failure mode.

Key generalizable lesson for the skill: NineToothed generated kernels ARE stride-aware when the argument rank matches the declared `Tensor(rank)`. The failure is caused by the rank mismatch (2-D view passed to rank-1 kernel), not by non-contiguity itself.

## Minimal Fix Options (evaluated)

- (a) Wrapper-level guard: validate `input.ndim == 1` (or `.flatten()`, which copies for non-contiguous views, as the silu wrapper already does) and document the copy semantics. Smallest, correct, costs one copy.
- (b) Rank-matched kernel: declare `Tensor(2)`; the generated launch then passes both strides and handles the non-contiguous view natively without a copy. Best performance, larger change.
- (c) Framework-level: add a launch-time assertion that argument ndim equals declared ndim, turning the silent wrong-answer into an explicit error. Out of scope for a minimal operator patch, but worth documenting.

Selected minimal fix: (a) for wrapper repair tasks; (b) when the task explicitly requires non-contiguous support without a copy.

## Verification Plan

Re-run on the GPU environment:

```python
out_fixed = add_fixed(a, b)
assert torch.allclose(out_fixed, a + b)
```

## Verification (closed loop)

Re-run on RTX 4090, torch 2.9.1+cu128, Triton 3.5.1, NineToothed 0.25.0 (2026-07-02):

Command:

```shell
python gpu-session/verify_layout_fix.py
```

Result:

```text
view shape (256, 256) stride (512, 2) contiguous False
[repro ] rank-1 kernel allclose=False (expect False), column-0-only correct=True (expect True)
[fix a ] flatten-guard allclose=True (expect True)
[fix b ] rank-2 kernel allclose=True (expect True)
[sanity] contiguous rank-2 allclose=True (expect True)
```

The reproduction confirms the root cause precisely: with the rank-1 kernel, only column 0 of the output is correct (the 256 elements addressed by `i * stride(0)`), everything else is garbage. Both fixes verify:

- Fix (a) `flatten()` guard: correct; costs one copy for non-contiguous input.
- Fix (b) rank-2 kernel (`Tensor(2)`, `tile((1, BLOCK_SIZE))`): correct on the same stepped view with no input copy, and correct on contiguous input.

Status: closed. Root cause traced to source, minimal fix implemented in two variants, both re-verified on GPU.

## Benchmark

Command:

```shell
Not run for this layout task.
```

Result:

```text
Not applicable. The layout task is a correctness/diagnosis case.
```

## Unsupported Cases

The tested public examples add wrapper should be treated as contiguous-only unless fixed. Unsupported or unproven cases include stepped views, transposed views, negative strides, overlapping storage, and arbitrary storage offsets.
