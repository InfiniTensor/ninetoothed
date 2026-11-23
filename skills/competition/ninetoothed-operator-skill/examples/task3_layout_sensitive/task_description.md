# T3: Layout-Sensitive GELU on Non-Contiguous Input

## Task Description

- **Operator type**: Layout-sensitive elementwise — GELU activation
- **Semantics**: `GELU(x) ≈ 0.5x(1 + tanh(0.7979(x + 0.044715x³)))` (tanh approximation)
- **Inputs**: X (M,N) float32 in 4 layout variants: contiguous, transposed (.T), sliced ([::2,:]), offset ([:,1:])
- **Output**: Y (M,N) float32 — identical logical result regardless of input layout
- **Implementation pattern**: Pattern A — 1D flatten tile for layout-agnostic elementwise access
- **Implementation file**: `operator_impl.py` (ninetoothed.make)

## Agent Execution Summary

- Read SKILL.md Phase 1-4, references/index.md Pattern A and Common Pitfall "Assuming contiguous input"
- Key insight: 1D flatten tiling + stride metadata from PyTorch tensor → Triton backend handles layout
- Arrangement: tile((BLOCK_SIZE,)) on both X and Y (flattened)
- Application: x_cubed = X³; inner = √(2/π) * (X + 0.044715 * x_cubed); Y = 0.5 * X * (1 + tanh(inner))
- Files created: operator_impl.py, test_correctness.py, test_layout_compare.py, benchmark.py

## Correctness Test

**Command**: `pytest examples/task3_layout_sensitive/test_correctness.py examples/task3_layout_sensitive/test_layout_compare.py -v`

**Coverage**: 14 parameterized cases + 2 cross-layout consistency tests. Four layout variants × multiple shapes including (2,2) minimum and (257,257) non-aligned.

**Status**: ✅ 12/12 PASSED — Tesla T4, 7.7s。`ntl.tanh()` 不存在，用 exp 手动实现 tanh。常量必须内联（JIT 无法捕获模块级变量）。

## Benchmark

**Command**: `python examples/task3_layout_sensitive/benchmark.py`

**Configurations**: contiguous, transposed, sliced(stride=2) × (4096,4096). Baseline: PyTorch `nn.GELU(approximate='tanh')`.

**Status**: ✅ 已完成 — contiguous 1.00x 持平 PyTorch，transposed 4.35x 性能恶化，直接证明布局敏感性。

## Failure Diagnosis

5 known failure modes documented:
1. Non-contiguous input returns contiguous results → kernel silently copies → check IR for strided load/store
2. Transposed input crash → misaligned address → check pointer arithmetic
3. Sliced input wrong values → ignores stride → verify stride metadata propagation
4. Offset input crash → missing storage_offset → add offset to base pointer
5. Wrong results at (257,257) → BLOCK_SIZE boundary → add boundary guard
