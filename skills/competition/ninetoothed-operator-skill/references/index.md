# NineToothed Operator Development — Reference Index

## 1. Repository Map

| Repository | URL | Purpose |
|---|---|---|
| **ninetoothed** | <https://github.com/InfiniTensor/ninetoothed> | Core DSL and compiler for defining GPU kernels via the arrange-and-apply paradigm. Source of truth for `Tensor`, `block_size()`, `ninetoothed.make()`, and `ninetoothed.language` primitives. |
| **ninetoothed-examples** | <https://github.com/InfiniTensor/ninetoothed-examples> | Reference operator implementations (elementwise, broadcast, reduction, matmul, softmax). Primary source of arrangement/application patterns. |
| **ntops** | <https://github.com/InfiniTensor/ntops> | Official operator library — production-quality implementations. Useful for comparing pattern choices and verifying performance baselines. |
| **InfiniCore** | <https://github.com/InfiniTensor/InfiniCore> | Core runtime and infrastructure libraries. Consult when diagnosing AOT build issues or inspecting generated Triton IR. |

### Key Files in the Main Repo

| File / Directory | Purpose |
|---|---|
| `README.md` | Project overview, arrange-and-apply paradigm explanation, minimal matmul walkthrough. Read first when onboarding. |
| `CONTRIBUTING.md` | PR rules, branch naming conventions, code style requirements, and pytest expectations. Must be read before submitting any PR. |
| `.githooks/` | Pre-commit hooks that run style checking and linting. Install them before making commits. |
| `scripts/check_contributing_style.py` | CI-style checker that validates branch names, commit messages, and file permissions against CONTRIBUTING.md rules. Run before opening a PR. |

---

## 2. Arrangement / Application Patterns

### Pattern A — Elementwise (1D Flatten Tile)

**When to consider:** The operator maps each input element to exactly one output element with no inter-element dependence. Examples: ReLU, GELU, elementwise add/mul, scaling.

**Arrangement strategy:** Flatten the tensor to 1D and tile with a single `BLOCK_SIZE`. For 2D inputs, use `X.tile((BLOCK_SIZE,))` where `BLOCK_SIZE` covers a contiguous chunk of the flattened element stream. This approach is layout-agnostic — strides are handled by the Triton backend via tensor metadata.

**Application strategy:** Direct elementwise math (`+`, `-`, `*`, `/`, `ntl.exp()`, `ntl.tanh()`, `ntl.maximum()`) inside the application function. No special boundary handling is needed for the elementwise case when using 1D flatten tiling.

**Key traps:**
- When the total element count is not evenly divisible by `BLOCK_SIZE`, the last tile reads a partial block. Verify that the generated Triton code includes a boundary mask (`mask=(offs < numel)`).
- For operators with multiple inputs, all inputs must share the same total element count or use broadcast (see Pattern B).

**Reference file:** `ninetoothed-examples/examples/task3_layout_sensitive/operator_impl.py` (GELU with 1D flatten tile).

---

### Pattern B — Elementwise with Broadcast

**When to consider:** The operator combines tensors of different ranks or shapes where one tensor's dimension broadcasts to match another's. Examples: vector-plus-scalar, matrix-plus-vector (row broadcast), masked add where a smaller tensor and/or mask tile broadcasts.

**Arrangement strategy:** Tile the larger tensor(s) at full rank (e.g., `A.tile((BLOCK_M, BLOCK_N))`). Tile the smaller broadcasting tensor at its own rank (e.g., `B.tile((BLOCK_N,))`), then call `.expand((BLOCK_M, -1))` to broadcast along the missing dimension. The `-1` dimension auto-matches the corresponding dimension of the tiled output.

**Application strategy:** After expand, all tensors have the same tiled shape, so application code can treat them as if they were same-shaped. Arithmetic blending (e.g., `A + B * mask` where mask acts as 0/1) is a common pattern.

**Key traps:**
- The `expand()` axis must match the broadcast dimension exactly. If the broadcasting tensor has shape `(N,)` and needs to broadcast to `(M, N)`, call `B_tiled.expand((BLOCK_M, -1))`. Using `-1` on the wrong axis is a frequent source of silent shape mismatch.
- Broadcasting a 0-D tensor (scalar): tile it as a size-1 tensor and expand.
- When combining broadcast with non-contiguous inputs, verify that the expand does not assume contiguous layout.

**Reference file:** `ninetoothed-examples/examples/task1_elementwise_broadcast/operator_impl.py` (masked add broadcast with `.expand((BLOCK_M, -1))`).

---

### Pattern C — Reduction Along an Axis

**When to consider:** The operator reduces one or more dimensions of the input (e.g., sum, max, min along a specific axis). Examples: row-wise sum, global max, reduce mean.

**Arrangement strategy:** Tile the input along all non-reduction axes at full size, and along the reduction axis with `BLOCK_SIZE`. For example, a row-wise reduction of a `(M, N)` matrix uses `X.tile((BLOCK_M, BLOCK_N))` where each tile processes `BLOCK_M` rows and `BLOCK_N` columns. The arrangement must account for the fact that `N` may not be evenly divisible by `BLOCK_N` — the last block along the reduction axis is partial.

**Application strategy:** Perform the reduction within each tile (e.g., `ntl.max(X, dim=1)`, `ntl.sum(X, dim=1)`). If the reduction requires cross-block merging (e.g., the reduction axis spans multiple tiles), the arrangement must handle the merge, typically by a second kernel pass or by using an online algorithm within a single pass.

**Key traps:**
- Boundary handling is **required** when the reduction dimension is not evenly divisible by `BLOCK_SIZE`. Without a boundary mask, the last partial block reads past the tensor boundary, causing either incorrect results (silent data corruption read) or a CUDA out-of-bounds error.
- Unlike elementwise patterns, a partial block in a reduction does not trivially produce correct partial results — the `ntl.max()` or `ntl.sum()` over a partially-filled tile must be masked.

**Reference file:** `ninetoothed-examples/examples/task2_reduction_block/operator_impl.py` (tiled softmax with per-tile max and sum). For an example of what happens without boundary handling, see: `ninetoothed-examples/examples/task4_benchmark_debug/buggy_operator.py`.

---

### Pattern D — Block / Tiled Multi-Pass

**When to consider:** The operator requires multiple passes over each tile (e.g., max-subtract then exp then sum then divide), or needs an online/numerically-stable algorithm within a block. Examples: softmax, layer norm, online softmax for long sequences.

**Arrangement strategy:** Same 2D tiling as Pattern C (tile along both the batch and reduction axes). The key difference is that the application function performs multiple sequential operations per tile. For softmax in particular, the per-tile computation follows the numerically-stable max-subtraction pattern: compute row max, subtract, exponentiate, sum, and divide.

**Application strategy:** Within the application function, chain the operations in the correct order: `max -> subtract -> exp -> sum -> divide`. For online softmax (where the reduction axis is longer than `BLOCK_SIZE`), the application must compute local max and sum, then the arrangement must handle merging across tiles with rescaling: `sum *= exp(old_max - new_max)`.

**Key traps:**
- Numerical stability: always subtract the per-row max *before* exponentiation to prevent `exp()` overflow. This applies to any softmax or softmax-like operator.
- When `N` is much larger than `BLOCK_N`, a naive single-tile softmax produces incorrect results because the denominator only covers one tile's contributions. Either use an online two-pass algorithm or ensure cross-block reduction merging.
- `keepdims=True` is important for correct broadcasting when subtracting the max from the original tensor.

**Reference file:** `ninetoothed-examples/examples/task2_reduction_block/operator_impl.py` (tiled softmax with max-subtraction and numerical stability).

---

### Pattern E — Matmul (2D Tile with Expand + Squeeze)

**When to consider:** The operator is a dense matrix multiplication or a variant (e.g., batched matmul, masked matmul). Examples: `C = A @ B`, batched matmul for attention scores.

**Arrangement strategy:** Use nested tensors with a 2D tile. Tile `A` along the M and K dimensions: `A.tile((BLOCK_M, BLOCK_K))`. Tile `B` along the K and N dimensions: `B.tile((BLOCK_K, BLOCK_N))`. Accumulate into `C.tile((BLOCK_M, BLOCK_N))`. The pattern uses `.expand()` on the accumulation dimension and `.dtype.squeeze()` on the nested-tensor dimension to collapse the dtype nesting created by expand.

**Application strategy:** Initialize an accumulator with `ntl.zeros(shape, dtype=...)`, then call `ntl.dot(acc, A_tiled, B_tiled)` within the application. After the dot product chain, the `.dtype.squeeze()` call in the arrangement removes the extra dimension from the dtype nesting.

**Key traps:**
- The dimension passed to `.dtype.squeeze()` must match the nested-tensor dimension created by expand. Squeezing the wrong dimension produces incorrect dtype-level shapes.
- When the K dimension is not evenly divisible by `BLOCK_K`, the last tile along K is partial. Boundary masking is needed.
- For batched matmul, add a batch dimension to the tile (e.g., `BLOCK_B`).

**Reference file:** `ninetoothed-examples/main_repo/README.md` (matmul walkthrough example). See also: `SKILL.md` DSL Quick Reference entry for `.dtype.squeeze()`.

---

## 3. Testing Patterns

### Correctness Test Template

```python
"""
Correctness Test for <operator_name>.

Tests:
  - Normal input shapes
  - Boundary: minimum size
  - Boundary: non-aligned (dim not divisible by BLOCK_SIZE)
  - Broadcast edge (if applicable)
  - Non-contiguous input
  - dtype variants
  - Extreme values
"""

import torch
import pytest

# Adjust import to match your operator module location
from operator_impl import your_operator


def pytorch_reference(*args):
    """PyTorch reference implementation matching operator semantics exactly."""
    # TODO: implement reference
    raise NotImplementedError


@pytest.mark.parametrize("shape", [
    (1024, 1024),  # Normal
    (1, 1),        # Minimum boundary
    (1024, 257),   # Non-aligned (N not divisible by typical BLOCK_SIZE)
])
def test_correctness_normal_boundary(shape):
    """Normal and boundary shape configurations."""
    M, N = shape
    device = "cuda"
    X = torch.randn((M, N), dtype=torch.float32, device=device)

    expected = pytorch_reference(X)
    output = torch.empty_like(X)
    your_operator(X, output)

    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-3)


def test_non_contiguous():
    """At least one non-contiguous input case (transposed, sliced, or stride-offset)."""
    device = "cuda"
    # Example: transposed input
    M, N = 256, 512
    X_base = torch.randn((N, M), dtype=torch.float32, device=device)
    X = X_base.T  # non-contiguous view

    expected = pytorch_reference(X)
    output = torch.empty_like(X)
    your_operator(X, output)

    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-3)


def test_float16():
    """float16 variant — use relaxed tolerance."""
    device = "cuda"
    M, N = 256, 256
    X = torch.randn((M, N), dtype=torch.float16, device=device)

    expected = pytorch_reference(X.float()).half()
    output = torch.empty((M, N), dtype=torch.float16, device=device)
    your_operator(X, output)

    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("extreme", [
    "large_pos",
    "large_neg",
    "mixed",
    "zeros",
])
def test_extreme_values(extreme):
    """Very large, very small, negative, and zero inputs."""
    device = "cuda"
    M, N = 128, 256
    if extreme == "large_pos":
        X = torch.full((M, N), 1e4, dtype=torch.float32, device=device)
    elif extreme == "large_neg":
        X = torch.full((M, N), -1e4, dtype=torch.float32, device=device)
    elif extreme == "mixed":
        X = torch.randn((M, N), dtype=torch.float32, device=device) * 1e4
    else:
        X = torch.zeros((M, N), dtype=torch.float32, device=device)

    expected = pytorch_reference(X)
    output = torch.empty_like(X)
    your_operator(X, output)

    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains inf"
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-3)
```

### Required Test Coverage Checklist

Every operator correctness test MUST include all seven categories below:

| # | Category | Description | Example shapes |
|---|---|---|---|
| 1 | **Normal input** | Typical representative shape with standard dtype (float32). | `(1024, 1024)`, `(256, 256)` |
| 2 | **Boundary: minimum size** | Smallest meaningful size to verify edge-case tile logic. | `(1,)`, `(1, 1)` |
| 3 | **Boundary: non-aligned** | Dimension not evenly divisible by `BLOCK_SIZE` to test boundary masking. | `(1024, 257)`, `(100, 100)` |
| 4 | **Broadcast edge** | A dimension of size 1 broadcasting to a large dimension (if operator uses broadcast). | `(1024, 1)` with `(1,)` vector |
| 5 | **Non-contiguous input** | At least one case where input strides differ from contiguous layout (transposed, sliced, or offset). | `.T`, `[::2, :]`, `[:, 1:]` |
| 6 | **dtype variants** | float32 (required) and float16 (if the operator is performance-sensitive). Use relaxed tolerance for float16. | float32 at `1e-5/1e-3`; float16 at `1e-3/1e-2` |
| 7 | **Extreme values** | Very large positive/negative, mixed large magnitudes, and zero values. Verify no NaN or inf. | `1e4`, `-1e4`, `torch.randn * 1e4`, `zeros` |

---

## 4. Benchmark Patterns

### Minimum Benchmark Record Template

Every benchmark must record all of the following fields in its output or accompanying report:

```
================================================================================
Operator: <operator_name>
PyTorch:  <torch.__version__>
CUDA:     <torch.version.cuda>
GPU:      <torch.cuda.get_device_name(0)>
Warmup:   <X> runs  |  Timed: <Y> runs
================================================================================

Shape            NineToothed (ms)   PyTorch (ms)        Ratio
--------------------------------------------------------------------------------
(1024, 1024)     0.1234             0.1100              1.12
(2048, 2048)     0.4567             0.4200              1.09
...

================================================================================
Conclusion:
  - <interpretation of results — e.g., "kernel is 1.1x slower than PyTorch for
    contiguous inputs, bottleneck appears to be memory bandwidth">
================================================================================
```

### Seven Required Fields

| # | Field | What to record |
|---|---|---|
| 1 | **Baseline** | PyTorch reference function used for comparison (e.g., `torch.nn.functional.softmax`, `torch.where`, `nn.GELU`). |
| 2 | **Input sizes** | Exact shape(s) tested. Minimum 3 configurations covering small, medium, and large. |
| 3 | **dtype** | float32 (primary) and float16 (if applicable). |
| 4 | **Layout** | Contiguous and at least one non-contiguous layout (transposed, sliced, or offset). |
| 5 | **Runs** | Number of warmup iterations (typically 50--100) and timed iterations (typically 200--1000). Both must be stated. |
| 6 | **Hardware** | GPU model name, CUDA version, PyTorch version, and (if known) Triton and NineToothed versions. |
| 7 | **Conclusion** | Written interpretation of the results: is the kernel competitive? What is the bottleneck? Any shape-dependent performance cliffs? |

### Reference Benchmark Files

- `ninetoothed-examples/examples/task1_elementwise_broadcast/benchmark.py` — elementwise broadcast with contiguous and non-contiguous layout variants.
- `ninetoothed-examples/examples/task2_reduction_block/benchmark.py` — multi-shape softmax benchmark.
- `ninetoothed-examples/examples/task3_layout_sensitive/benchmark.py` — layout-comparison benchmark (contiguous vs transposed vs sliced).
- `ninetoothed-examples/examples/task4_benchmark_debug/benchmark_multisize.py` — detailed multi-shape benchmark with BLOCK_SIZE sensitivity notes.

---

## 5. Common Pitfalls

### Pitfall 1: BLOCK_SIZE Boundary (Missing Partial-Block Handling)

- **Symptom:** Tests pass for shapes where dimensions divide `BLOCK_SIZE` evenly (e.g., 1024 with `BLOCK_SIZE=256`) but fail for shapes that do not (e.g., 257 with `BLOCK_SIZE=256`). Failure mode is either a CUDA out-of-bounds error or silently incorrect tail values.
- **Root cause:** The application assumes every tile is full-sized. The last partial block reads/writes beyond the tensor allocation.
- **Fix:** Add a boundary mask in the generated Triton code (`mask=(offs < N)`) or adjust the arrangement to clamp the last tile size. Test every operator with at least one non-aligned shape.
- **Reference:** `ninetoothed-examples/examples/task4_benchmark_debug/buggy_operator.py` (deliberate boundary bug) and its diagnosis at `ninetoothed-examples/examples/task4_benchmark_debug/diagnosis_record.md`.

### Pitfall 2: Wrong Expand Axis in Broadcast

- **Symptom:** Silent shape mismatch, incorrect output values, or a NineToothed compilation error about incompatible tensor shapes.
- **Root cause:** The `.expand(dim0, dim1)` call broadcasts along the wrong axis. For example, expanding a `(BLOCK_N,)` tile with `.expand((BLOCK_N, -1))` instead of `.expand((BLOCK_M, -1))`.
- **Fix:** Verify that the expand target dimension matches the tiled output dimension. For a 2D tile `(BLOCK_M, BLOCK_N)`, a `(BLOCK_N,)` input should expand to `(BLOCK_M, -1)`. Use `-1` for the auto-matched dimension.
- **Reference:** `ninetoothed-examples/examples/task1_elementwise_broadcast/operator_impl.py` — correct expand usage.

### Pitfall 3: Missing Squeeze on Nested-Tensor Dtype

- **Symptom:** NineToothed compilation error about dtype nesting dimensions not matching, or a kernel that produces a tensor with unexpected shape/dtype.
- **Root cause:** In matmul-like patterns (Pattern E), `.expand()` on the accumulation dimension creates a nested tensor dimension in the dtype. This nesting must be removed with `.dtype.squeeze(dim)`. If the squeeze dimension is wrong or missing, the kernel cannot compile.
- **Fix:** Identify which expand dimension created the nesting and squeeze that exact dimension. The matmul walkthrough in the main repo's README provides the canonical pattern.
- **Reference:** `ninetoothed-examples/main_repo/README.md` (matmul example); `SKILL.md` DSL Quick Reference.

### Pitfall 4: Assuming Contiguous Layout

- **Symptom:** Operator works correctly on contiguous tensors but produces wrong results on transposed, sliced, or offset inputs. May silently produce incorrect values without raising an error.
- **Root cause:** The arrangement assumes standard row-major memory layout. In a transposed tensor, the stride along dim 0 is 1 and dim 1 is the leading dimension — the kernel reads values from the wrong memory locations.
- **Fix:** Verify that the operator correctly reads stride metadata. The 1D flatten tiling approach (Pattern A) is naturally layout-agnostic. For multi-dimensional tiles, test every operator with at least one non-contiguous shape.
- **Reference:** `ninetoothed-examples/examples/task3_layout_sensitive/operator_impl.py` (1D flatten GELU — layout-agnostic by design).

### Pitfall 5: Float16 Overflow in Intermediate Computation

- **Symptom:** NaN or inf values in output when using float16, even though float32 works correctly. Typically occurs in softmax (exp of large values) or in chain multiplications.
- **Root cause:** float16 has limited dynamic range (~6e-5 to ~6e4). Values above 65,504 become inf. Exponentiating a value of 12 or higher produces inf in float16.
- **Fix:** For numerically sensitive operators (softmax, exp-based activations), accumulate in float32 and cast down at the end. Use the max-subtraction trick even more aggressively for float16.
- **Reference:** `ninetoothed-examples/examples/task2_reduction_block/operator_impl.py` (numerically stable softmax with max-subtraction).

### Pitfall 6: Autotuning Too Slow for Development Iteration

- **Symptom:** Each kernel compilation takes 30--60 seconds during development because `block_size()` triggers autotuning on every invocation.
- **Root cause:** Using `BLOCK_SIZE = block_size()` during fast development iteration. Autotuning searches over multiple BLOCK_SIZE values and compiles each variant, even though the developer is still debugging correctness.
- **Fix:** During development, use a fixed integer (e.g., `BLOCK_SIZE = 256`) to disable autotuning. Switch to `BLOCK_SIZE = block_size()` only for the final benchmark pass. The same applies to other tunable parameters.
- **Reference:** `SKILL.md` Phase 3, step 6.

---

## 6. Code Style Checklist

### Pre-Submission Checks

Run the following in order before opening a PR:

1. **ruff format** — Auto-format all Python files:
   ```bash
   ruff format skills/competition/ninetoothed-operator-skill/
   ```

2. **ruff check** — Lint all Python files:
   ```bash
   ruff check skills/competition/ninetoothed-operator-skill/
   ```
   Fix all errors before proceeding.

3. **check_contributing_style.py** — Validate branch names, commit messages, and file permissions against CONTRIBUTING.md rules:
   ```bash
   python scripts/check_contributing_style.py
   ```

4. **pytest** — Run the structural validation tests (no GPU needed):
   ```bash
   pytest skills/competition/ninetoothed-operator-skill/tests/test_skill_structure.py -v
   ```
   Then run all correctness tests (GPU required):
   ```bash
   pytest skills/competition/ninetoothed-operator-skill/examples/task1_elementwise_broadcast/test_correctness.py -v
   pytest skills/competition/ninetoothed-operator-skill/examples/task2_reduction_block/test_correctness.py -v
   pytest skills/competition/ninetoothed-operator-skill/examples/task3_layout_sensitive/test_correctness.py -v
   ```

### Branch Naming

Format: `2026-spring-<githubid>-t3-1-1`

Rules:
- All lowercase, kebab-case.
- Year (2026) comes **first**, before the season.
- Maximum 50 characters total.
- GitHub ID in the middle position.
- Problem ID (`t3-1-1`) at the end in lowercase.

Examples:
- `2026-spring-junkai-kay-t3-1-1` (valid)
- `2026-spring-alice12345-t3-1-1` (valid)

### PR Title

Format: `Add 2026 spring T3-1-1 skill submission for <githubid>`

Examples:
- `Add 2026 spring T3-1-1 skill submission for junkai-kay`
- `Add 2026 spring T3-1-1 skill submission for alice12345`

---

*This reference index is part of the `ninetoothed-operator-skill` package. For the core workflow, see `SKILL.md`. For PR and compliance rules, see `CONTRIBUTING.md` and `HONOR_CODE.md`.*
