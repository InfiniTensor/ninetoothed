# T4 Sub-Task 4C: API Mismatch Bug — Diagnosis Record

## Bug Description

The buggy kernel (`buggy_operator.py`) uses `ntl.max(input_row, dim=1, keepdims=True)` and
`ntl.sum(exp_x, dim=1, keepdims=True)`. These API signatures do not exist in NineToothed 0.26.

## Actual Diagnosis (from Colab session, 2026-07-11)

### Step 1 — Reproduce

**Command:**
```bash
cd /content/ninetoothed
python -c "
import torch
import sys
sys.path.insert(0, 'skills/competition/ninetoothed-operator-skill/examples/task4_benchmark_debug')
from buggy_operator import tiled_softmax_buggy
X = torch.randn((4, 256), dtype=torch.float32, device='cuda')
Y = torch.empty_like(X)
tiled_softmax_buggy(X, Y)
"
```

**Symptom (actual GPU output, 2026-07-12, Tesla T4):**
```
CompilationError: at 10:12:
    x_max = triton.language.max(..., dim=1, keepdims=True)
            ^
TypeError("max() got an unexpected keyword argument 'dim'")
```

The kernel compiles to Triton IR but the underlying `triton.language.max()` rejects
the `dim` and `keepdims` keyword arguments.

### Step 2 — Isolate

| Test | Result |
|---|---|
| T2 correct version (test_correctness.py) | 13/13 PASSED |
| T2 buggy version (buggy_operator.py) | CompilationError — all shapes fail |
| `ntl.max(x)` without args | ✅ works |
| `ntl.max(x, dim=1)` | ❌ CompilationError |
| `ntl.sum(x)` without args | ✅ works |
| `ntl.sum(x, dim=1)` | ❌ CompilationError |
| `ntl.tanh(x)` | ❌ function does not exist |

### Step 3 — Diagnose

**Method:** Ran `dir(ninetoothed.language)` in Colab to inspect the actual API surface.

```python
import ninetoothed.language as ntl
print([x for x in dir(ntl) if not x.startswith('_')])
# Output: ['LANGUAGE', 'Symbol', 'ast', 'attribute', 'call', 'libdevice']
```

Only 6 attributes — no `tanh`, and `max`/`sum` do not appear at all at
the module level (they exist as methods on the internal `LANGUAGE` object
with different signatures).

Then read the official test file:
```bash
cat /content/ninetoothed/tests/test_softmax.py
```

Found the correct pattern:
- `@ninetoothed.jit` decorator (not `ninetoothed.make`)
- `Symbol("BLOCK_SIZE", constexpr=True)` (not `block_size()` or `meta=True`)
- `Tensor(2, other=float("-inf")).tile((1, BLOCK_SIZE))` (one row per tile)
- `ntl.max(input_row)` — no arguments, reduces entire tile
- `ntl.sum(numerator)` — no arguments, reduces entire tile
- Pass `BLOCK_SIZE=input.shape[-1]` at call time

### Step 4 — Minimal Fix

**Fix applied to** `examples/task2_reduction_block/operator_impl.py`:

**Before (broken):**
```python
x_max = ntl.max(input_row, dim=1, keepdims=True)
exp_sum = ntl.sum(exp_x, dim=1, keepdims=True)
```

**After (fixed):**
```python
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
# Tile as (1, BLOCK_SIZE) — one row → ntl.max() gives row-wise max
x_max = ntl.max(input_row)
exp_sum = ntl.sum(numerator)
```

And call with:
```python
_kernel(X, Y, BLOCK_SIZE=X.shape[-1])
```

### Step 5 — Re-verify

**Command:**
```bash
pytest skills/competition/ninetoothed-operator-skill/examples/task2_reduction_block/test_correctness.py -v
```

**Result:**
```
13 passed in 8.04s
```

All tests pass including extreme values and non-contiguous input.

## Closure

| Item | Detail |
|---|---|
| Symptom | `triton.compiler.errors.CompilationError` on all inputs |
| Root cause | `ntl.max()` and `ntl.sum()` in NineToothed 0.26 do not support `dim=`/`keepdims=` arguments; they reduce the entire tile to a scalar |
| Minimal fix | Remove `dim=` and `keepdims=` arguments; use `(1, BLOCK_SIZE)` tiling so full-tile reduction = row-wise reduction |
| Re-run result | 13/13 PASSED (8.04s) |
| Lessons | (1) Always inspect `dir(ninetoothed.language)` to know the actual API surface; (2) Read the official test files in the repo — `tests/test_softmax.py` contains the canonical pattern; (3) Do NOT install `ninetoothed` from PyPI — use `git clone` + `pip install -e .` |

## Additional: T3 GELU API Diagnosis

A similar diagnosis was performed for T3 GELU:
- `ntl.tanh()`: does not exist → replaced with `tanh(z) = (ntl.exp(2z) - 1) / (ntl.exp(2z) + 1)`
- Module-level constants (`math.sqrt(2/π)`): not captured by JIT → inlined as float literals
