# Upstream synchronization on 2026-11-03

This maintenance merges official master `4295e9c767805b619418f9ab8d0692f186772b1b`
into PR #220, starting from `88cda0e5a59714f49f2833d4a6eb37a88a8fda65`.
Both remote branch tips were checked before the merge and again after testing.
The PR is not merged into the official repository by this maintenance.

## Conflict resolution

The two conflicting hunks are in `_decompose_matmul_store` in
`src/ninetoothed/compiler/passes.py`:

- Preserve upstream ASCII numeric dimension recognition and materialize numeric
  K, including zero, as an `index/int64` `arith.constant`.
- Retain the PR's `shape.dim(-1)` bound for compound arranged dimensions, and
  retain direct operands for symbolic dimensions.
- Retain integer `0` for the accumulator initializer, recursive provenance
  derivation, and lane/operand/term/prefix result mappings.

`tests/test_interpreter_matmul_bounds.py` covers K `0`, `1`, `4`, `k`, `2*k`,
and `2*2`, for float32 and int32, through both the decomposition pass and the
CUDA/Triton pipelines interpreted on the CPU. Integer checks use exact equality
with inputs above `2**24`. Additional cases cover numeric K obtained from the
RHS or an explicit operation attribute when LHS shape information is absent.
The existing tiled-dot regression now checks the constant tile bound of four.

The remaining merged changes are upstream's AscendC and BangC baseline support.
The automatic emitter merge retains the PR's signed division, value-coordinate,
and broadcast address/mask changes.

## Validation

Tested on macOS arm64, Python 3.12.13, NumPy 2.5.4, SymPy 1.14.0,
pytest 9.2.0 and Ruff 0.17.3. The main test environment has no Torch or Triton.
The tested Git `src` subtree is `5a0973c47affe02a70915b1c577b78b12262c0d5`;
the `tests` subtree is `4466c9d755f6efadb9136a9c531c3b93083fa509`.

```text
python scripts/run_cpu_tests.py --junitxml cpu-tests.xml
986 passed, 17 deselected, 1 warning in 42.04s
```

The warning is NumPy 2.5 deprecating direct assignment to `array.shape` in an
existing geometry-revalidation test. The 17 excluded tests require actual GPU
execution; they are not passes.

Independent selections (overlap with the full CPU run; do not add counts):

```text
tests/test_interpreter_matmul_bounds.py
16 passed in 0.74s

tests/test_interpreter_applications.py
tests/test_interpreter_provenance.py
tests/test_ssa_pass_pipeline.py
98 passed in 8.68s

tests/test_ascendc_backend.py::test_ascendc_source_and_launch_contract
tests/test_bangc_backend.py::test_bangc_scalar_source_contract
tests/test_backend_registry.py
26 passed in 1.42s
```

With Torch 2.15.0 loaded from a separate dependency directory, the additional
frontend/nested-shape selection passed (`22 passed in 1.46s`). Torch CPU adapter
tests and the mocked BangC current-device lookup passed (`10 passed, 1 deselected,
1 warning in 1.16s`); the deselected test requires a real CUDA tensor and the
warning describes Torch's sparse invariant checks. Neither selection executes
GPU code or installs Torch into the main NumPy-only environment.

The initial frontend selection had `1 failed, 21 passed` because cache identity
imports Torch. The first Sphinx attempt also failed on the debugging module's
Torch import. These environment failures are recorded separately from the
successful checks after supplying the optional dependency. Sphinx 9.1.1
`sphinx-build -W -b html` then completed successfully.

Ruff lint, formatting, the contributing style checker and `git diff --check`
passed. A normal wheel built successfully; all 106 Python source files matched
the tested source byte-for-byte. Outside the checkout, isolated `python -I`
imports, the debug demo and its exported standalone differential replay passed
without Torch/Triton. The demo's deliberately injected fault is expected.

## Boundaries

Zero K produces correct zero outputs and no inner-loop events. Its four inner
checkpoint mappings report `Reference inputs must be complete rank-two tiles.`
because the existing empty-input capture loses the rank information. Loading
the old PR head's pass implementation reproduces exactly the same diagnostic;
this maintenance does not change the debugger runtime. Nonzero K checkpoint
comparisons have no mapping issues.

No NVIDIA GPU, Ascend NPU or Cambricon MLU execution, device JIT/AOT, performance
measurement, or full hardware `pytest` suite was performed. Source-generation
checks do not establish hardware correctness. Historical hardware results apply
only to their recorded commits. GitHub CI is checked separately after pushing;
local results do not establish that remote checks passed.
