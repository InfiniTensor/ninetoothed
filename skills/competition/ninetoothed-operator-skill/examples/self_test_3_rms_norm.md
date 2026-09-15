# Self-test Task 3: Layout-sensitive — RMS Norm (Non-contiguous Input)

## 1. Task description

- **Operator type:** Normalization (RMS Norm, no learnable weight)
- **Inputs:** `input` — 2D float32 tensor, possibly non-contiguous
- **Outputs:** `output` — 2D float32 tensor, same shape as input
- **Formula:** `output[i] = input[i] / sqrt(mean(input[i]^2) + eps)`
- **Shape constraints:** 2D input `(n_rows, n_cols)`, n_cols must be power-of-2
- **Dtype constraints:** float32
- **Layout constraints:** input may be non-contiguous (transposed, row-sliced, col-sliced)
- **Boundary cases:** small rows (n_cols=64), large rows (n_cols=512), stride > 1

## 2. Agent execution summary

**Files inspected:**
- `ninetoothed-examples/ops/ninetoothed/kernels/rms_norm.py` — reduction pattern reference
- `ntops` rms_norm implementation — multi-pass loop, `Tensor(ndim, other=0)` pattern
- NineToothed `Symbol` API for constexpr parameters

**Implementation pattern selected:**
- `Symbol("BLOCK_SIZE", constexpr=True)` for explicit BLOCK_SIZE at call time
- `Tensor(2).tile((1, BLOCK_SIZE))` arrangement — one program per row
- `ntl.cast(input, ntl.float32)` inside application for dtype safety
- `input.shape[1]` for denominator (tile dimension, equals n_cols when BLOCK_SIZE=n_cols)
- Non-contiguous inputs handled via `.contiguous()` before kernel dispatch

**Files added/modified:**
- `ops/rms_norm.py` — new kernel implementation

## 3. Correctness test

```bash
pytest tests/test_rms_norm_noncontiguous.py -v
```

Result:
```
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0
rootdir: /content/ninetoothed-operator-skill
configfile: pyproject.toml

tests/test_rms_norm_noncontiguous.py::test_rms_norm_contiguous[shape0] PASSED
tests/test_rms_norm_noncontiguous.py::test_rms_norm_contiguous[shape1] PASSED
tests/test_rms_norm_noncontiguous.py::test_rms_norm_contiguous[shape2] PASSED
tests/test_rms_norm_noncontiguous.py::test_rms_norm_transposed PASSED
tests/test_rms_norm_noncontiguous.py::test_rms_norm_row_slice PASSED
tests/test_rms_norm_noncontiguous.py::test_rms_norm_col_slice PASSED
tests/test_rms_norm_noncontiguous.py::test_rms_norm_rejects_dynamic_eps PASSED
tests/test_rms_norm_noncontiguous.py::test_rms_norm_copy_overhead PASSED

7 correctness tests passed; 1 copy-overhead benchmark passed
```

Hardware: Google Colab T4 GPU, NineToothed 0.26.0, Python 3.12.13

## 4. Failure diagnosis (8-attempt record)

This task required 8 attempts to fix. The failure history is documented as a
diagnostic case study for SKILL.md Step 8.

| Attempt | Approach | Error | Root cause |
|---------|----------|-------|------------|
| 1 | `eps` as `Tensor(0)`, direct assign | `IncompatibleTypeErrorImpl: pointer<fp32>` | `Tensor(0)` scalar → pointer type in Triton IR |
| 2 | `Tensor(0)` + slice assign `output[...]=` | Same pointer error | Same root cause |
| 3 | Remove `eps`, closure capture of `BLOCK_SIZE` | `NameError: BLOCK_SIZE not defined` | NineToothed AST doesn't capture Symbol closures |
| 4 | `BLOCK_SIZE` as application default param | `NameError` in generated code | Symbol not resolvable at definition time |
| 5 | `BLOCK_SIZE` as regular application param | `BLOCK_SIZE` undefined at kernel call | NineToothed doesn't forward unknown params to application |
| 6 | Division by `BLOCK_SIZE` via closure | Same `NameError` | Symbol closure still fails |
| 7 | `input.shape[-1]` in application | `Triton cannot evaluate (1, BLOCK_SIZE)[-1]` | Triton constexpr tuples don't support negative indexing |
| **8** | **`input.shape[1]` in application** | **PASSED** | Positive index `[1]` on `(1, BLOCK_SIZE)` is valid constexpr |

**Key lessons recorded in SKILL.md:**
- `Tensor(0)` rank-0 scalars always become `pointer<fp32>` in generated Triton IR — never use for arithmetic arguments; use Python literals instead
- NineToothed's application function does NOT capture closures from outer scope — only parameters and module-level imports are visible
- Triton constexpr tuple indexing requires non-negative indices — `shape[-1]` fails, `shape[1]` works when the tile is 2D `(1, BLOCK_SIZE)`
- Non-contiguous inputs: call `.contiguous()` before kernel dispatch; NineToothed v0.26.0 does not auto-handle arbitrary strides in the arrangement
