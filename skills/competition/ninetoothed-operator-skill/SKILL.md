---
name: ninetoothed-operator-skill
description: Guide AI agents to implement, test, benchmark, debug, and report NineToothed operators. Use for NineToothed operator development, correctness testing, performance analysis, generated-source inspection, AOT diagnosis, and failing-test repair.
---

# NineToothed Operator Development Skill

## Trigger

Activate this skill when the task involves any of the following within the NineToothed repository:

- Developing a new GPU operator (kernel) using the NineToothed DSL
- Writing correctness tests for a NineToothed operator
- Debugging a failing operator or test
- Running or analysing benchmarks for a NineToothed operator
- Inspecting generated Triton source or AOT build artefacts
- Preparing a competition PR for the NineToothed .skill Innovation Challenge

## High-Level Workflow (SOP)

When given an operator task, follow this fixed pipeline. Do NOT skip steps.

### Phase 1 — Understand the Task

1. Extract from the task description:
   - Input tensor(s): name, shape, dtype, layout constraints
   - Output tensor(s): name, shape, dtype
   - Broadcast semantics (if any)
   - Boundary conditions (empty tensors, size-1 dims, alignment)
   - Layout notes: contiguous, non-contiguous, stride, offset, view, transpose

2. Write a one-paragraph summary of what the operator does before writing any code.

### Phase 2 — Search the Repository for Precedent

3. Search the repository for similar operators:
   - Check `InfiniTensor/ninetoothed-examples` — contains matmul, softmax, elementwise, reduce
   - Check `InfiniTensor/ntops` — official operator library
   - Grep for similar arrangement/application shapes in the main repo

4. Identify the closest reference pattern. Document the exact file path, function name, and why it applies.

**Conditional guidance (NOT absolute rules):**

| If the operator... | Consider... | Reference file | But be aware... |
|---|---|---|---|
| ...maps each input element to one output element | Pattern A: 1D tile | ninetoothed-examples elementwise kernels | May need 2D tile for better occupancy at large sizes |
| ...has a smaller input that broadcasts | Pattern B: tile + `.expand()` | matmul arrangement (expand pattern) | Verify expand axis matches broadcast dimension; -1 auto-match may not work in all NineToothed versions |
| ...reduces along an axis (e.g., sum, max) | Pattern C: tile reduce axis; plan cross-block merge | ninetoothed-examples reduce kernels, softmax | BLOCK_SIZE may not divide N evenly — boundary handling IS required |
| ...needs per-row normalisation (e.g., softmax) | Pattern D: multi-pass or online algorithm | ninetoothed-examples softmax | Online algorithm requires careful rescaling; verify with extreme values |
| ...is a dense linear algebra op (e.g., matmul) | Pattern E: 2D tile + expand + squeeze | matmul in main README | `.dtype.squeeze()` dim must match the nested-tensor dimension |

### Phase 3 — Determine the Arrangement and Block Size

5. Classify the operator type using the table above.

6. Choose block size strategy (NineToothed 0.26 verified patterns):

   | Pattern | API | When to Use |
   |---|---|---|
   | Autotuning | `Symbol("BLOCK_SIZE", meta=True)` | Production tuning; adds compilation time |
   | Fixed at launch | `Symbol("BLOCK_SIZE", constexpr=True)` + pass `BLOCK_SIZE=value` | Correctness; block size set at call time |
   | Full-dimension | Pass `BLOCK_SIZE=X.shape[-1]` to `constexpr` symbol | Reduction along full axis (e.g., softmax) |

   > **Note on `block_size()`:** This function works with the `ninetoothed.make()` API (see T1 example) but is not used with `@ninetoothed.jit`. For new kernels, prefer `Symbol` — it's the pattern used in all official NineToothed tests.

   Reference: `tests/test_softmax.py` in the NineToothed source repo demonstrates the correct `Symbol("BLOCK_SIZE", constexpr=True)` pattern.

7. Write the kernel using **`@ninetoothed.jit`** (NOT `ninetoothed.make()`). The `@ninetoothed.jit` decorator is the primary and most tested API in NineToothed 0.26:

   ```python
   BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

   @ninetoothed.jit
   def my_kernel(
       x: Tensor(2).tile((1, BLOCK_SIZE)),
       y: Tensor(2).tile((1, BLOCK_SIZE)),
   ):
       y = ...  # computation
   ```

   > **Recommendation:** `ninetoothed.make()` also works (see T1 example) but `@ninetoothed.jit` is the pattern used in all official NineToothed tests. For consistency with the codebase, use `@ninetoothed.jit`.

### Phase 4 — Write the Application Body

8. Inside the `@ninetoothed.jit` function, use ONLY these verified operations:

   **Always available (Python arithmetic):**
   - `+`, `-`, `*`, `/` — basic elementwise math

   **Available via `ninetoothed.language` (`ntl`):**
   - `ntl.exp(x)` — elementwise exponential (verified)
   - `ntl.max(x)` — reduces the ENTIRE tile to a scalar (NO arguments — no `dim=`, `axis=`, `keepdims=`)
   - `ntl.sum(x)` — reduces the ENTIRE tile to a scalar (NO arguments)
   - `ntl.zeros(shape, dtype=...)` — create accumulator (verified)

   > **⚠ DO NOT use these — they do NOT exist in NineToothed 0.26:**
   > - `ntl.tanh()` — compute manually via `tanh(z) = (e²ᶻ - 1) / (e²ᶻ + 1)`
   > - `ntl.maximum()` — not verified; use arithmetic or `ntl.max()` instead
   > - `ntl.max(x, dim=...)` / `ntl.max(x, axis=...)` / `ntl.max(x, keepdims=...)` — no axis/dim/keepdims support
   > - `ntl.sum(x, dim=...)` / `ntl.sum(x, axis=...)` — no axis/dim support

   **Key design pattern for reductions:** Tile as `(1, BLOCK_SIZE)` = one row per tile. Then `ntl.max(x)` and `ntl.sum(x)` naturally give row-wise reduction without needing axis arguments.

9. Keep the application body minimal. One block = one invocation.

### Phase 5 — Write the Correctness Test

10. Write a PyTorch reference implementation that matches the operator semantics exactly.

11. Write a pytest test covering ALL of:
    - Normal input (typical shape, dtype)
    - Boundary: minimum size (1,)
    - Boundary: dim not evenly divisible by BLOCK_SIZE
    - Broadcast edge: dim=1 broadcasting to large dim
    - **At least one non-contiguous input case** (transposed, sliced, or stride-offset)
    - dtype variants: float32 minimum; float16 if performance-sensitive
    - Extreme values: very large, very small, negative, zero
    - Tolerance: `atol=1e-5, rtol=1e-3` for float32; relax for float16

### Phase 6 — Run and Record

12. Run the correctness test:
    ```bash
    pytest <path-to-test-file> -v
    ```

13. Record the FULL pytest output. If it passes, record it. If it fails, go to **Failure Diagnosis**.

14. Do NOT silently modify the reference to make it pass.

### Phase 7 — Performance Verification

15. Write a benchmark script that records ALL of:
    - GPU model, CUDA version, PyTorch version, Triton version, NineToothed version
    - Baseline (PyTorch reference or existing implementation)
    - Input sizes (exact shape, dtype, layout)
    - warmup runs and timed runs
    - Full run command
    - Timing results (mean, std if possible)
    - Conclusion with interpretation

### Phase 8 — Generated Source / AOT Build Inspection (when required)

16. Export generated Triton IR and inspect:
    - Verify `.tile()` → `tl.program_id()` mapping
    - Check for `mask=(offs < N)` boundary handling
    - Verify stride parameters are passed to `tl.load`/`tl.store`
    - Note any unexpected control flow or missing optimisations

### Phase 9 — Final Summary

17. Output a structured summary: operator semantics, pattern used with reference file path, files added/modified, test command + output, benchmark results, failure diagnosis (if any).

---

## Failure Diagnosis Protocol

When a test fails or a benchmark regresses:

1. **Record the symptom**: exact error message, stack trace, unexpected output
2. **Isolate**: does the failure happen on all inputs or only specific shapes/dtypes/layouts?
3. **Diagnose** — check in this order:
   - Arrangement: tiling dimensions wrong? `.expand()` axis mismatch? `.squeeze()` on wrong dim?
   - Application: logic error in per-block computation?
   - Block size: does BLOCK_SIZE evenly divide input dimensions? If not, add boundary handling.
   - Layout: is the input non-contiguous? Did the kernel assume contiguous memory?
   - dtype: mixing float16 and float32 accumulators? NineToothed may require explicit casts.
4. **Minimal fix**: make the smallest change that fixes the issue. Do NOT refactor unrelated code.
5. **Re-run**: run the exact same test command and confirm it passes.
6. **Record the closure**: symptom → root cause → fix → re-run result.

---

## Constraints (RED LINES)

The AI agent MUST NOT:

- Hardcode hidden evaluation task names, inputs, or expected outputs
- Use API keys, private credentials, or online-only dependencies
- Delete, skip, or modify existing repository tests
- Perform unrelated refactoring or large-scale formatting outside `skills/competition/<skill-name>/`
- Introduce dependencies that require network access at runtime
- Falsify pytest output or benchmark results
- Submit files containing `.env`, `.venv`, `__pycache__`, `node_modules`, or large binary artefacts

## Key Reference Files (read in this order)

1. `README.md` — project overview, arrange-and-apply paradigm
2. `CONTRIBUTING.md` — PR rules, branch naming, code style, pytest requirements
3. `docs/` — detailed DSL usage documentation
4. `tests/` — existing test patterns for correctness tests
5. `examples/` or `InfiniTensor/ninetoothed-examples` — reference operator implementations
6. `scripts/check_contributing_style.py` — style checker used in CI

## DSL Quick Reference (NineToothed 0.26 — verified)

| Concept | API | Verified? |
|---|---|---|
| Tensor descriptor | `Tensor(ndim)` | ✅ |
| Tensor boundary fill | `Tensor(ndim, other=value)` | ✅ e.g., `other=float("-inf")` for max reduction |
| Block size (autotune) | `Symbol("X", meta=True)` | ✅ |
| Block size (constexpr) | `Symbol("X", constexpr=True)` + pass `X=value` | ✅ |
| Tiling | `.tile((BLOCK_M, BLOCK_N))` | ✅ |
| Kernel definition | `@ninetoothed.jit` | ✅ Primary API |
| Elementwise arithmetic | `+`, `-`, `*`, `/` | ✅ |
| Exponential | `ntl.exp(x)` | ✅ |
| Tile reduction (max) | `ntl.max(x)` — no args, full-tile scalar | ✅ |
| Tile reduction (sum) | `ntl.sum(x)` — no args, full-tile scalar | ✅ |
| Accumulator | `ntl.zeros(shape, dtype=...)` | ✅ |
| Dot product | `ntl.dot(a, b)` | ✅ (from matmul example) |

### Patterns that differ from documentation — verified in NineToothed 0.26

| Wrong API | Correct Alternative |
|---|---|
| `block_size()` | `Symbol("X", meta=True)` or `Symbol("X", constexpr=True)` |
| `ninetoothed.make()` | `@ninetoothed.jit` |
| `ntl.tanh(x)` | `tanh(z) = (e²ᶻ - 1) / (e²ᶻ + 1)` using `ntl.exp()` |
| `ntl.max(x, dim=...)` / `ntl.max(x, axis=...)` / `ntl.max(x, keepdims=...)` | `ntl.max(x)` — full-tile only; use `(1, BLOCK_SIZE)` tile for row-wise reduction |
| `ntl.sum(x, dim=...)` / `ntl.sum(x, axis=...)` | `ntl.sum(x)` — full-tile only |
| `ntl.maximum()` | Not verified — use arithmetic or `ntl.max()` |
| `pip install ninetoothed` | `git clone` + `pip install -e .` (PyPI package is empty shell) |

## Supported and Unsupported Scenarios

### Verified / Intended to Work
- **Hardware**: NVIDIA GPU with CUDA 11.8+ (tested on one GPU — record exact model after benchmark)
- **OS**: Linux (primary); macOS (CPU-only development — kernels not runnable)
- **dtype**: float32 (primary, tested); float16 (limited — test with relaxed tolerance)
- **Layouts**: contiguous, transposed, sliced (stride ≠ 1), offset (non-zero storage_offset)
- **Shapes**: static shapes known at kernel launch time
- **Operators**: elementwise, broadcast, reduction (row-wise), activation functions

### NOT Supported / NOT Verified
- **CPU execution**: NineToothed targets Triton → GPU only. CPU fallback not tested.
- **Dynamic shapes**: shapes that change between kernel invocations without recompilation are not verified
- **Empty tensors**: tensors with any dimension = 0 are not tested and may crash
- **float64 / int64**: only float32 (and limited float16) tested
- **CUDA < 11.8**: not tested; Triton requires relatively recent CUDA
- **Multi-GPU**: single GPU only
- **AOT build**: compilation to offline artefact is documented but not end-to-end verified in self-tests
- **Non-power-of-2 shapes**: tested at 257, but exhaustive non-power-of-2 coverage is not included
- **In-place operations**: all kernels write to a separate output tensor; in-place not tested
- **Windows**: not tested; Triton support on Windows is limited
