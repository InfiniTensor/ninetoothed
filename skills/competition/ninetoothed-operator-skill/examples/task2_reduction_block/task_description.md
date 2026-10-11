# T2: Tiled Softmax with Numerical Stability

## Task Description

- **Operator type**: Reduction / block-based — row-wise softmax with tiling
- **Semantics**: `Y[i,j] = exp(X[i,j] - max_i) / sum_j(exp(X[i,j] - max_i))`, max-subtraction for numerical stability
- **Inputs**: X (M,N) float32
- **Output**: Y (M,N) float32 (row sums = 1.0)
- **Implementation pattern**: Pattern D — 2D tile with within-tile max-subtract → exp → sum → normalize
- **Implementation file**: `operator_impl.py` (ninetoothed.make)

## Agent Execution Summary

- Read SKILL.md Phase 1-4 and references/index.md Pattern D
- Searched ninetoothed-examples softmax for online algorithm reference
- Arrangement: tile((BLOCK_M, BLOCK_N))
- Application: x_max = ntl.max(X, dim=1, keepdims=True); x_shifted = X - x_max; exp_x = ntl.exp(x_shifted); Y = exp_x / ntl.sum(exp_x, dim=1, keepdims=True)
- Files created: operator_impl.py, test_correctness.py, benchmark.py

## Correctness Test

**Command**: `pytest examples/task2_reduction_block/test_correctness.py -v`

**Coverage**: 12 cases — 6 shapes (including N=32768 long sequence and N=257 non-aligned), 4 extreme value ranges (±1e4), row-sum invariant, uniform-input test, non-contiguous input.

**Status**: ✅ 13/13 PASSED — Tesla T4, 8.0s. 使用 `@ninetoothed.jit` + `Symbol(constexpr=True)` + `(1, BLOCK_SIZE)` tile + `ntl.max/sum` 无参数归约。

## Benchmark

**Command**: `python examples/task2_reduction_block/benchmark.py`

**Configurations**: 6 shapes from (256,256) to (4096,1024). Baseline: PyTorch `F.softmax(X, dim=-1)`.

**Status**: ✅ 已完成 — 除单行短序列 (1,4096) 外，其余均不逊于 PyTorch。大 batch 下 0.61-0.89x。

## Failure Diagnosis

6 known failure modes documented:
1. NaN/inf for large inputs → add max-subtraction
2. Row sums ≠ 1.0 → cross-block merge error
3. BLOCK_SIZE boundary (N=257) → add boundary mask
4. N=1 degenerate case → verify div-by-zero not possible (exp(0)=1)
5. Cross-block max not propagated → check variable scoping
6. float16 overflow → use float32 accumulator
