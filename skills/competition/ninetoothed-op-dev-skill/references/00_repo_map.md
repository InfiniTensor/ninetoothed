# 00 — Repository map

> Paths use **`<repo-root>/...`** where `--repo-root` is the NineToothed clone
> (directory containing `src/ninetoothed/`).  
> Optional examples use **`<examples-root>/...`** via `--examples-root`
> (or `NINETOOTHED_EXAMPLES_ROOT`).

## 1. Layout overview

```text
<repo-root>/                        # --repo-root .
  src/ninetoothed/                  # DSL, codegen, JIT, AOT, debugging
  tests/                            # Official correctness + generation/AOT tests
  docs/source/                      # Sphinx: basics, build (AOT), API
  skills/competition/ninetoothed-op-dev-skill/   # this skill (when installed in-fork)

<examples-root>/                    # optional --examples-root
  ops/ninetoothed/kernels/          # Canonical arrange-and-apply kernels
  ops/triton/kernels/               # Triton baselines
  tests/                            # Cross-op correctness + benchmark markers
```

## 2. Core concepts → source files

| Concept | Primary modules | Notes |
|---------|-----------------|-------|
| `Tensor`, tiling | `<repo-root>/src/ninetoothed/tensor.py` | Rank, `tile`, `expand`, `ravel`, stride views |
| `Symbol`, `block_size` | `<repo-root>/src/ninetoothed/symbol.py` | `meta=True` → autotuning; `constexpr=True` → fixed at launch |
| `ninetoothed.language` (`ntl`) | `<repo-root>/src/ninetoothed/language.py` | `dot`, `max`, `sum`, `exp`, `load`, `atomic_add`, … |
| `ninetoothed.make` | `<repo-root>/src/ninetoothed/make.py` | arrange-and-apply → kernel |
| `@ninetoothed.jit` | `<repo-root>/src/ninetoothed/jit.py` | Inline kernel functions (see `tests/test_add.py`) |
| Code generation | `<repo-root>/src/ninetoothed/generation.py` | Generated Triton/C++ source; `kernel._source` |
| AOT build | `<repo-root>/src/ninetoothed/build.py`, `aot.py` | Ahead-of-time `.so`; see `docs/source/build.rst` |
| Autotuner | `<repo-root>/src/ninetoothed/auto_tuner.py` | `test_auto_tuner.py`, `test_generation.py` |
| Debugging / viz | `<repo-root>/src/ninetoothed/debugging.py`, `visualization.py` | `test_debugging.py` |
| dtypes | `<repo-root>/src/ninetoothed/dtype.py` | `float16`, `bfloat16`, `float32`, … |

Public API surface: `<repo-root>/src/ninetoothed/__init__.py` exports `make`, `jit`, `Tensor`, `Symbol`, `block_size`, `build`, dtypes.

## 3. `<repo-root>/tests/` index (agent search order)

| File | Use for |
|------|---------|
| `test_add.py` | Elementwise add, **`@ninetoothed.jit`** style |
| `test_expand.py`, `test_pow.py` | Broadcast / elementwise |
| `test_softmax.py` | Row softmax, numerical stability (`max` before `exp`) |
| `test_matmul.py`, `test_addmm.py` | `make` + reduction dot pattern |
| `test_max_pool2d.py` | 4D block reduce / window |
| `test_conv2d.py` | Spatial / padding |
| `test_data_ptr.py` | `data_ptr`, `atomic_add`, storage-level ops |
| `test_clone.py` | **stride/offset** via `offsets`, `stride`, manual `ntl.load` |
| `test_getitem.py` | Indexing / views |
| `test_generation.py` | Read **generated source** from `kernel._source` |
| `test_aot.py`, `test_aot_auto_tuning.py` | AOT compile + load |
| `test_auto_tuner.py` | Tuning configs |
| `test_debugging.py` | Debug hooks |
| `test_naming.py` | Light smoke / naming (good first pytest) |
| `conftest.py` | Per-test deterministic seeds |
| `utils.py` | `get_available_devices()` → `cuda` (and optional `mlu`) |

## 4. `<examples-root>/` index (optional)

| Path | Use for |
|------|---------|
| `ops/ninetoothed/kernels/add.py` | Minimal `make` elementwise template |
| `ops/ninetoothed/kernels/softmax.py` | Reduction |
| `ops/ninetoothed/kernels/max_pool2d.py` | Block/window max pool |
| `ops/ninetoothed/kernels/mm.py`, `bmm.py` | Matmul family |
| `ops/ninetoothed/kernels/conv2d.py` | Conv |
| `ops/ninetoothed/torch.py` | Wiring kernels to torch tensors |
| `ops/triton/kernels/*` | Baseline implementations |
| `tests/test_ops.py` | Correctness across ops |
| `tests/test_benchmarks.py` | `@pytest.mark.benchmark` |
| `README.md` | `pytest`, `pytest -m benchmark`, autotuning disable |
| `bench.py` | `assert_match`, plotting helpers |

## 5. Commands

```bash
REPO="<repo-root>"          # --repo-root
EXAMPLES="<examples-root>"  # optional --examples-root
```

### NineToothed repo tests

```bash
cd "$REPO"
pytest tests/test_naming.py -v --tb=short
pytest tests/test_add.py -v --tb=short
pytest tests/test_softmax.py -v --tb=short
pytest tests/test_data_ptr.py -v --tb=short
pytest tests/test_generation.py -v --tb=short
pytest tests/test_aot.py -v --tb=short   # slower; may need build toolchain
```

### Examples (optional)

```bash
cd "$EXAMPLES"
pip install -e .    # ⚠️ user action; not auto-run by skill
pytest -v --tb=short
pytest -m benchmark -k TestMM
```

### Skill installability checks (inside the fork)

```bash
cd "$REPO"
pytest skills/competition/ninetoothed-op-dev-skill/tests -q
python skills/competition/ninetoothed-op-dev-skill/scripts/quick_validate.py
python skills/competition/ninetoothed-op-dev-skill/scripts/env_check.py --repo-root .
```

## 6. Two implementation styles (do not mix blindly)

### Style A — `@ninetoothed.jit` (compact, used in some tests)

See `<repo-root>/tests/test_add.py`.

### Style B — `make(arrangement, application, tensors)` (examples + README matmul)

See `<examples-root>/ops/ninetoothed/kernels/add.py` and `<repo-root>/README.md` matmul.

**Agent rule:** Find the closest existing file for your operator family and **match its style**.

## 7. Code style conventions

- Tests: `pytest`, `@pytest.mark.parametrize`, `torch.allclose`, `get_available_devices()`
- Kernel body assigns via `output = expr  # noqa: F841` pattern
- Examples: `Symbol(..., meta=True)` for autotuning; use `constexpr=True` + launch arg to disable for fast dev
- Seeds: automatic per module/test via `conftest.py`

## 8. Common misconceptions

| Wrong | Right |
|-------|-------|
| Invent `ntl.*` ops without checking `language.py` | Grep `<repo-root>/src/ninetoothed/language.py` or copy from nearest test |
| Patch compiler core by default | Prefer user/examples targets or skill `examples/` trajectories |
| Assume CPU tests | `get_available_devices()` skips CPU; CUDA required for most tests |
| Run full `pytest` + autotuning on first try | Disable autotuning or run a single file |
| `nvcc` always available | AOT may need CUDA toolkit; declare unsupported if missing |

## 9. Agent file search order

1. Task type → table in §3 or §4  
2. Open matching `<examples-root>/ops/ninetoothed/kernels/<op>.py` if `--examples-root` is set  
3. Else open `<repo-root>/tests/test_<op>.py`  
4. Read `conftest.py` + `utils.py` for device/dtype patterns  
5. For perf/AOT → `test_generation.py`, `test_aot.py`, `docs/source/build.rst`
