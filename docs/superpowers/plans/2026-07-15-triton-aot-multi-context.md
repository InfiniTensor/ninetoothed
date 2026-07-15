# Triton AOT Multi-Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make fresh and reloaded Triton AOT handles safe to reuse across CUDA contexts.

**Architecture:** Link a NineToothed-owned C++ companion beside Triton's generated AOT sources. The companion swaps generated module/function globals from a mutex-protected `CUcontext` state map around every host launch and marks the cached binary with a launcher schema.

**Tech Stack:** Python 3.10+, ctypes, CUDA Driver API, NVCC/C++17, pytest, Ruff.

---

### Task 1: Add CPU-Visible Context Guard Contracts

**Files:**
- Modify: `src/ninetoothed/backends/materializers/triton.py`
- Create: `tests/test_triton_aot_materializer.py`

- [ ] **Step 1: Write failing symbol-discovery and source-generation tests**

Add representative Triton generated C fixtures containing `CUmodule`,
`CUfunction`, and `load_*` declarations. Assert that production helpers return
the low-level kernel names, produce companion source containing
`cuCtxGetCurrent`, `std::mutex`, a `CUcontext` state map, and the stable
`ninetoothed_triton_enter`/`leave` exports. Add a malformed fixture that must
raise `ValueError`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest tests/test_triton_aot_materializer.py -q
```

Expected: FAIL because the symbol-discovery and companion-generation helpers
do not exist.

- [ ] **Step 3: Implement strict symbol discovery and companion generation**

In `triton.py`, add private helpers that:

```python
def _triton_aot_kernel_names(sources: tuple[Path, ...]) -> tuple[str, ...]:
    """Return low-level kernels with matching module, function, and loader symbols."""


def _triton_context_guard_source(kernel_names: tuple[str, ...]) -> str:
    """Generate the context guard linked beside Triton's AOT launchers."""
```

The generated C++ must maintain arrays for every discovered low-level kernel,
cache state by `CUcontext`, and hold a mutex from `enter` through `leave`.

- [ ] **Step 4: Add failing wrapper-order tests**

Use fake ctypes-style exports to assert these event sequences:

```python
["enter", "kernel", "leave"]
["enter", "kernel", "leave"]  # kernel returns a CUDA error
["enter", "kernel", "leave"]  # kernel raises
["enter"]                      # enter returns a CUDA error
```

- [ ] **Step 5: Route fresh and reloaded AOT calls through the guard**

Create one loader helper used by `_aot_materialize` and
`TritonMaterializer.load_built_artifact`. Set ctypes signatures for the kernel,
enter, and leave exports. Update `_aot_wrapper` so a successful `enter` is
paired with `leave` in `finally`, while preserving existing ABI binding and
`KernelLaunchError` behavior.

- [ ] **Step 6: Verify focused tests GREEN**

Run:

```bash
pytest tests/test_triton_aot_materializer.py -q
```

Expected: all tests pass without a CUDA device.

### Task 2: Link And Cache The Guarded Binary

**Files:**
- Modify: `src/ninetoothed/backends/materializers/triton.py`
- Modify: `tests/test_triton_aot_materializer.py`

- [ ] **Step 1: Write failing link-input and stale-cache tests**

Patch `subprocess.run` and manifest reads to assert that the generated
companion `.cu` file is present in the final NVCC command and that a manifest
without the current launcher schema triggers `_compile_aot_library` even when
the `.so` exists.

- [ ] **Step 2: Run the focused tests and verify RED**

Run `pytest tests/test_triton_aot_materializer.py -q` and confirm failures are
caused by the missing companion link input and schema check.

- [ ] **Step 3: Implement guarded linking and schema invalidation**

After `triton.tools.compile`, discover all low-level symbols, write the
companion into the build temporary directory, and include it in NVCC inputs.
Add a Triton AOT launcher schema to the manifest and recompile stale cached
binaries while holding the existing artifact lock.

- [ ] **Step 4: Verify focused and adjacent tests**

Run:

```bash
pytest tests/test_triton_aot_materializer.py tests/test_built_artifact_reload.py tests/test_compiler_cache_runtime.py -q
```

Expected: CPU-visible tests pass; CUDA-only tests may skip when CUDA is absent.

### Task 3: Add And Run The Real Multi-Context Regression

**Files:**
- Modify: `tests/test_built_artifact_reload.py`

- [ ] **Step 1: Add an explicit Triton AOT multi-context test**

Compile one uniquely named float32 add artifact with `backend="triton"`. Run
the fresh handle on devices `(0, 1, 0)` and a reloaded handle on `(1, 0, 1)`,
synchronizing and checking output after every launch. Skip unless two GPUs with
matching compute capability are available.

- [ ] **Step 2: Verify the regression is RED on the unguarded implementation**

Before production changes, run the test on `ssh nvidia` and retain the expected
second-context `KernelLaunchError(400)` output.

- [ ] **Step 3: Verify the regression and original selector GREEN**

Run on `ssh nvidia` with `accelerator-dev/nvidia:latest`:

```bash
pytest tests/test_built_artifact_reload.py::test_triton_aot_handle_is_reusable_across_cuda_contexts
pytest 'tests/test_aot.py::test_add[True-45327-dtype0-bf16-cuda]'
```

Expected: both commands pass.

### Task 4: Repository Validation And Publication

**Files:**
- Modify only files already listed by this plan.

- [ ] **Step 1: Run repository-required checks**

Run:

```bash
ruff format --check
ruff check
python scripts/check_contributing_style.py
pytest
```

Expected: all checks pass, with environment-specific GPU skips documented.

- [ ] **Step 2: Review the final diff**

Confirm there are no caller/backend routing changes, unrelated refactors, or
unintended generated artifacts.

- [ ] **Step 3: Commit, push, and open a draft PR**

Use an imperative commit title without punctuation, push
`fix-triton-aot-multi-context`, and create a draft PR targeting
`ssa-compiler-with-multibackend`. Include the full required pytest output and
the remote multi-GPU commands in the PR description.
