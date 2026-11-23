# T1: Masked Add with Broadcast

## Task Description

- **Operator type**: Elementwise with broadcast and mask
- **Semantics**: `C[i,j] = mask[i,j] ? (A[i,j] + B[j]) : A[i,j]` where B broadcasts along rows
- **Inputs**: A (M,N) float32, B (N,) float32, mask (M,N) bool
- **Output**: C (M,N) float32
- **Implementation pattern**: Pattern B — 2D tile + `.expand()` broadcast
- **Implementation file**: `operator_impl.py` (ninetoothed.make with arrangement/application/tensors)

## Agent Execution Summary

- Read SKILL.md Phase 1-4 for workflow
- Searched ninetoothed-examples for elementwise and broadcast patterns
- Selected Pattern B: tile((BLOCK_M, BLOCK_N)) on A/mask/C, tile((BLOCK_N,)) + expand((BLOCK_M, -1)) on B
- Application: `C = A + B * mask` (arithmetic mask blend)
- Files created: operator_impl.py, test_correctness.py, benchmark.py

## Correctness Test

**Command**: `pytest examples/task1_elementwise_broadcast/test_correctness.py -v`

**Coverage**: 11 parameterized cases — normal shapes (256²–2048²), boundary (M=1, N=1, 257² non-aligned), mask patterns (all_true, all_false, checkerboard), non-contiguous (transposed A), float16 variant.

**Status**: ✅ 12/12 PASSED — Tesla T4, CUDA 12.8, PyTorch 2.11.0, NineToothed 0.26.0.

## Benchmark

**Command**: `python examples/task1_elementwise_broadcast/benchmark.py`

**Configurations**: 4 contiguous sizes + 1 non-contiguous (transposed). Baseline: PyTorch `torch.where(mask, A+B, A)`.

**Status**: ✅ 已完成 — 大矩阵 NineToothed 反超 PyTorch (0.44x at 4096²), 非连续输入 0.79x。

## Failure Diagnosis

4 known failure modes documented in references/index.md §5:
1. Wrong `.expand()` axis → shape mismatch → verify target dimensions
2. BLOCK_SIZE boundary → incorrect edges → add boundary check
3. Missing `.dtype.squeeze()` → type error → squeeze correct dim
4. Non-contiguous input → incorrect results → verify stride handling
