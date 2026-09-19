# Testing the CPU Reference Interpreter

This guide covers three things, ordered by how much hardware they need:

1. **CPU-only tests**: run the interpreter test suite with no GPU.
2. **GPU tests**: run the project's full suite once you have a CUDA server.
3. **CPU vs. GPU cross-validation**: the strongest check. Run the same
   application through the interpreter and through the real Triton/CUDA kernel,
   then compare the numbers.

---

## 1. Setup

### On a laptop / CI box with no GPU

The interpreter needs only NumPy. `ninetoothed` imports Triton lazily, so you can
install it without a working GPU toolchain:

```shell
git clone https://github.com/InfiniTensor/ninetoothed.git
cd ninetoothed

python -m venv .venv
source .venv/bin/activate
pip install -e .            # triton is declared as a dependency
pip install pytest numpy
```

> **macOS note.** There is no `triton` wheel for macOS, so `pip install -e .`
> will fail there. Install the source tree without pulling dependencies:
>
> ```shell
> pip install -e . --no-deps
> pip install numpy sympy pytest
> ```
>
> `import ninetoothed` and `from ninetoothed.interpret import interpret` both work
> this way, because `ninetoothed.language` resolves `libdevice` lazily and nothing
> on the interpreter path imports Triton.

### On a CUDA server

`git clone https://github.com/InfiniTensor/ninetoothed.git` gets you the upstream
repository, which does not contain the interpreter. That work is not pushed
there. Bring the checkout over instead:

```shell
# Option A: upload `ninetoothed-server.tar.gz` through the JupyterLab file
# browser (the ⬆ button), then:
cd /root/autodl-tmp
tar -xzf ninetoothed-server.tar.gz
cd ninetoothed
bash setup_on_server.sh          # installs, runs the GPU-free suite, prints the matrix

# Option B: copy from your laptop:
scp -P <port> ninetoothed-server.tar.gz root@<host>:/root/autodl-tmp/
```

Doing it by hand is the same:

```shell
cd /root/autodl-tmp/ninetoothed

python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your driver
pip install pytest numpy
```

Most rented PyTorch images already ship a matching `torch`/`triton` pair. In that
case skip the `torch` line and run `pip install -e . --no-deps`.

Verify the GPU is visible before going further:

```shell
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
```

### What the environment has to satisfy

`pyproject.toml` has the exact versions:

| Requirement | Declared |
| --- | --- |
| Python | `>=3.10` |
| Runtime deps | `triton>=3.0.0`, `sympy>=1.13.0`, `numpy>=1.26.4` |
| `debugging` extra | `torch>=2.4.0` (needed by the 28 test files that import torch) |

A stock PyTorch 2.4+ / Python 3.10–3.12 / CUDA 12.x image satisfies all of this,
because the PyTorch wheel brings a matching Triton (`torch 2.4 → triton 3.0`,
`2.5 → 3.1`, `2.6 → 3.2`). Do not pin Triton separately. `pip install -e .` sees
that the torch-provided version already satisfies `triton>=3.0.0` and leaves it
alone. If pip ever tries to move it, install the project with `--no-deps`.

### Which GPU, and does the model matter?

One GPU is enough. The interpreter and every test in this repository are
single-device, and there is no collective communication anywhere in the path.

The model matters for one thing: the compute architecture.

| GPU | Compute capability | Platform profile |
| --- | --- | --- |
| A100 / **A800** | `sm_80` | `nvidia-a100` |
| H100 | `sm_90` | `nvidia-h100` |

`A800` is the A100 with reduced NVLink bandwidth. The compute capability, the
memory and the generated Triton code are identical, so an A800 run reproduces an
A100 run exactly. If the grading environment is stated as A100, developing on an
A800 is safe, but re-run `cross_validate.py` on an actual A100 before submitting
if you want the artifact to say A100.

`resolve_target_context` does not probe the device. With no
`NINETOOTHED_PLATFORM` set, the profile is `generic` and the architecture is left
to Triton, which probes the device itself, so the default Triton backend works on
any GPU with no configuration:

```shell
python -c "from ninetoothed.targets import resolve_target_context as r; c = r(None); print(c.backend.value, c.platform.name, c.compute_arch)"
# triton generic None
```

The CUDA backend (`backend="cuda"`) is the exception: `generic` carries no
architecture and refuses to guess, so it must be told.

```shell
export NINETOOTHED_PLATFORM=nvidia-a100   # sm_80, matches A100 and A800
# or, equivalently, pass platform="nvidia-a100" / compute_arch="sm_80" explicitly
```

The `nvidia-a100` profile also declares `unsupported_capabilities={"dtype.fp8"}`.
That matches the project scope: `float8` is out of scope, and the interpreter
refuses it too.

---

## 2. CPU-only tests (no GPU required)

Most of the verification can happen before you rent a machine.

```shell
# The interpreter's own suite.
python -m pytest tests/test_interpret.py -v

# The GPU-free part of the existing suite.
python -m pytest \
    tests/test_ssa_application_lowering.py \
    tests/test_ssa_validation.py \
    tests/test_ssa_pass_pipeline.py \
    tests/test_ssa_first_backend_lowering.py \
    -q
```

Expected: `51 passed` for `test_interpret.py`, and `59 passed` for the four SSA
files (`110 passed` together).

### Everything that runs without a GPU

27 of the 47 test files import `torch` at module scope, so they cannot even be
collected on a laptop. Of the remaining 20, 19 need no device at all: 18 need only
NumPy, and `test_lowering_inference.py` additionally imports `triton.language`.
The last one, `test_ipynb.py`, needs `jupytext`. With `triton` installed, 248 of
the tests in the 19 files below pass:

```shell
python -m pytest -q \
    tests/test_backend_registry.py \
    tests/test_compiler_cache_runtime.py \
    tests/test_compiler_entrypoints.py \
    tests/test_emitter_boundaries.py \
    tests/test_eval.py \
    tests/test_getitem.py \
    tests/test_interpret.py \
    tests/test_ir_immutability.py \
    tests/test_kernel_ir.py \
    tests/test_layout_transfer_analysis.py \
    tests/test_lowering_inference.py \
    tests/test_materializer_registry.py \
    tests/test_naming.py \
    tests/test_ssa_application_lowering.py \
    tests/test_ssa_first_backend_lowering.py \
    tests/test_ssa_pass_pipeline.py \
    tests/test_ssa_validation.py \
    tests/test_target_profiles.py \
    tests/test_unsqueeze.py
```

Expected: `248 passed, 1 failed`. The single failure is
`test_compiler_cache_runtime.py::test_cuda_compiler_identity_is_part_of_compilation_cache_key`,
which asks the cache for the runtime CUDA architecture and so needs a GPU; it
passes on the server. Everything else is a real regression if it fails.

If pytest refuses to start with `ModuleNotFoundError: No module named 'torch'`,
your checkout predates the graceful-degradation change in `tests/conftest.py`.
Add the following to that file:

```python
try:
    import torch
except ModuleNotFoundError:
    torch = None
```

and guard the `torch.manual_seed(seed)` call in `_set_random_seed`.

### The whole suite, on the server

With `torch` and a CUDA device present, every file collects and the remaining 28
files run too. This is the broadest regression check:

```shell
python -m pytest -q --continue-on-collection-errors tests/
```

It compiles a real Triton kernel per test, so budget 20–40 minutes on a single
GPU. `-q` prints one character per test, so a lone `.` on a stalled-looking line
is progress, not a hang. To run it without babysitting:

```shell
nohup python -m pytest -q --continue-on-collection-errors tests/ > pytest.log 2>&1 &
tail -f pytest.log      # Ctrl+C leaves the tail, not the run
```

A clean result on one A800 (Python 3.12.3, CUDA 12, torch + triton preinstalled):

```
2 failed, 546 passed, 10 skipped in 1935.96s (0:32:15)
```

Both failures are missing optional third-party tools, not defects. Neither test
file imports the interpreter, and neither tool is a declared dependency:

| Failure | Cause | Fix |
| --- | --- | --- |
| `test_built_artifact_reload.py::test_aot_built_artifact_can_be_reloaded[tilelang-cuda]` | `ImportError: TileLang is required…` | `pip install tilelang` (large; optional) |
| `test_ipynb.py::test_ipynb[cuda]` | `FileNotFoundError: 'jupytext'` | `pip install jupytext` |

The 10 skips are hardware guards: `test_built_artifact_reload.py` and
`test_aot.py` skip their multi-device cases unless two or more CUDA devices are
visible, and `test_jagged.py` skips when jagged nested tensors are unavailable.

### What the suite actually checks

| Area | What is asserted |
| --- | --- |
| Semantics | Results match a NumPy reference for elementwise, broadcast, reduction, softmax, layernorm, matmul, transpose, integer and boolean ops |
| Masking | A masked-out lane never touches the buffer; the untouched tail keeps the caller's sentinel value |
| Bounds | An unmasked access outside the buffer raises instead of reading the wrong element |
| Layout | The resolved access map equals `ninetoothed.eval._eval` (the compiler's own ground truth) for 1-D, 2-D and nested tiles |
| Dtypes | Integer/bool results are bit-exact; `float32` is not silently widened; `bfloat16` is refused |
| Diagnostics | Failures name the offending SSA opcode and location; unknown symbols list the available ones |
| Tooling | Traces record program ids and mask counts; `compare_pipeline` is semantics-preserving; `compare_passes` names the first diverging pass and pins it on a program instance and a `mem.store`; reproduction snippets carry the SSA, data, shape, dtype and seed, and compile |

---

## 3. GPU tests

Once you have a CUDA machine, run the full suite:

```shell
python -m pytest tests/ -q
```

Then confirm the interpreter agrees with the real backend. The script below is
the cross-validation check: it compiles the application for the GPU, runs it, and
compares the result against the CPU interpretation of the same lowered program.

The repository already ships this as `cross_validate.py` in the root. Run it on
the server with `python cross_validate.py`. Its shape is:

```python
"""Compare a CPU interpretation against a real Triton/CUDA execution."""

import dataclasses

import numpy as np
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.interpret import compare_interpretations, interpret

ROWS, WIDTH, BLOCK = 3, 11, 16


def arrangement(x, out):
    return x.tile((1, BLOCK)), out.tile((1, BLOCK))


def application(x, out):
    shifted = x - ntl.max(x, axis=1)[:, None]
    numerator = ntl.exp(shifted)
    out = numerator / ntl.sum(numerator, axis=1)[:, None]  # noqa: F841


def main():
    # --- CPU reference, using the interpreter -----------------------------
    cpu_x = np.random.default_rng(0).random((ROWS, WIDTH), dtype=np.float32)
    cpu_out = np.zeros_like(cpu_x)

    reference = interpret(
        arrangement,
        application,
        tensors=(Tensor(2, other=float("-inf")), Tensor(2)),
        inputs=(cpu_x, cpu_out),
    )

    # --- GPU execution, using the same arrangement and application --------
    kernel = ninetoothed.make(
        arrangement,
        application,
        (Tensor(2, other=float("-inf")), Tensor(2)),
    )

    gpu_x = torch.from_numpy(cpu_x).cuda()
    gpu_out = torch.zeros((ROWS, WIDTH), dtype=torch.float32, device="cuda")
    kernel(gpu_x, gpu_out)

    gpu_result = gpu_out.cpu().numpy()

    # --- Compare, using the interpreter's own differ ----------------------
    # `Interpretation` is a dataclass, so the GPU output can be substituted
    # into a copy of the CPU run and the two diffed like any other pair.
    on_gpu = dataclasses.replace(reference, outputs={"out": gpu_result})
    diff = compare_interpretations(reference, on_gpu, label="cpu vs gpu")

    print(diff.render())  # per-output mismatch count and max error

    if not diff.matches:
        print(diff.to_json())  # for a CI log
        print(reference.render_trace(limit=40))
        raise SystemExit(1)

    print("MATCH: the GPU kernel agrees with the CPU reference")

    # Also compare against a NumPy reference, to catch a shared misunderstanding.
    shifted = cpu_x - cpu_x.max(axis=1, keepdims=True)
    numerator = np.exp(shifted)
    numpy_expected = numerator / numerator.sum(axis=1, keepdims=True)
    error = float(np.abs(reference.output("out") - numpy_expected).max())
    print("max |cpu - numpy| =", error)


if __name__ == "__main__":
    main()
```

Run it:

```shell
python cross_validate.py
```

A clean run prints `MATCH` and a `max |cpu - numpy|` around `1e-8`. The default
tolerance is `rtol=1e-3, atol=1e-3`, the project's standard for `float32`. Pass
`rtol=1e-6, atol=1e-6` to `compare_interpretations` to see how much margin there
is.

### Comparing a real kernel against saved outputs

If you already have outputs saved from a GPU run (or produced by PyTorch),
substitute them into a copy of the CPU interpretation and diff the two.
`Interpretation` is a dataclass, so `dataclasses.replace` does the job:

```python
import dataclasses

import numpy as np

from ninetoothed.interpret import compare_interpretations, interpret

cpu = interpret(arrangement, application, inputs=(x, out))

# `gpu_out.npy` was written on the server by the same application.
gpu = dataclasses.replace(cpu, outputs={"out": np.load("gpu_out.npy")})

diff = compare_interpretations(cpu, gpu, label="cpu vs gpu")
print(diff.render())  # per-output mismatch count, first indices, max error
print(diff.to_json())  # the same information, for CI logs
```

If the two disagree, `diff.outputs["out"].first_mismatches` gives the indices to
look at, and `result.render_trace()` on the CPU side shows what the interpreter
did at each step.

---

## 4. Validating a pass pipeline

A pass pipeline must be semantics preserving. The interpreter checks that
directly, with no GPU involved:

```python
from ninetoothed.interpret import compare_pipeline

diff = compare_pipeline(
    arrangement,
    application,
    tensors=(Tensor(2, other=float("-inf")), Tensor(2)),
    inputs=(x, out),
    pipeline=["ssa.canonicalize", "ssa.analyze_effects"],
    trace=True,
)

print(diff.render())

if not diff.matches:
    print(diff.minimal_reproduction())  # a runnable snippet for a bug report
```

### Finding *which* pass broke it

`compare_pipeline` answers "is the pipeline sound?". `compare_passes` answers
"which pass is at fault?". It applies the pipeline one pass at a time and
interprets the program at every cumulative prefix:

```python
from ninetoothed.interpret import compare_passes

diff = compare_passes(
    arrangement,
    application,
    tensors=(Tensor(2, other=float("-inf")), Tensor(2)),
    inputs=(x, out),
    # pipeline=None bisects the default pipeline of the backend;
    # pass a list of names to bisect a custom one instead.
)

print(diff.render())
```

```
pipeline diff: application pipeline
passes: ssa.canonicalize, ssa.analyze_effects, ssa.select_schedule,
        ssa.triton.optimize_schedule, ssa.decompose_linalg,
        ssa.validate_target_capabilities
result: MATCH
stages:
  [base ]  0 <frontend>
  [match]  1 ssa.canonicalize
  [match]  2 ssa.analyze_effects
  [match]  3 ssa.select_schedule
  [match]  4 ssa.triton.optimize_schedule
  [match]  5 ssa.decompose_linalg
  [match]  6 ssa.validate_target_capabilities
```

When a stage does diverge, `localize()` narrows it to a program instance and the
`mem.store` that produces the wrong value:

```python
localization = diff.localize()

print(localization.render())
```

```
first semantic difference introduced by `ssa.select_schedule`
  output      : out(1,)
  expected    : 2.0
  actual      : 3.0
  program id  : 0
  stores      :
    entry:2:mem.store
```

A stage the interpreter cannot execute (a pass it has not been taught yet) is
recorded with its error instead of aborting the scan, so a partially covered
pipeline still gives you the stages that do run.

### The reproduction

`diff.reproduction()` (or `diff.minimal_reproduction()` for the snippet) returns
everything a bug report needs: the executed SSA, the input data with its shapes
and dtypes, and the recorded random seed.

```python
from ninetoothed.interpret import build_reproduction, random_inputs

# Generate inputs from a seed so the failing run is reproducible...
x, out = random_inputs([((3, 11), "float32"), ((3, 11), "float32")], seed=17)

result = interpret(
    arrangement,
    application,
    inputs=(x, out),
    seed=17,  # ...and record it on the interpretation
)

reproduction = build_reproduction(result, label="softmax")
print(reproduction.seed)  # 17
print(reproduction.to_json())  # SSA + data + shape + dtype + seed
print(reproduction.render())  # a snippet that re-runs the case
```

---

## 5. Debugging a failing kernel

The workflow when a GPU kernel gives a wrong answer:

1. **Reproduce on the CPU first.** Shrink the input to something small (a few
   rows) and run the application through `interpret`. If it fails there, the bug
   is in the arrangement or the application, not in the backend.

2. **Inspect the layout.** Compare what the interpreter resolved against what
   the compiler reports:

   ```python
   import numpy as np
   from ninetoothed import Tensor
   from ninetoothed.eval import _eval
   from ninetoothed.interpret import access_mask, access_offsets

   offsets = access_offsets(result, "x")
   mask = access_mask(result, "x")
   print(np.where(mask, offsets, -1))  # interpreter
   print(_eval(arranged_x, subs))  # compiler, same layout expected
   ```

   They must be identical. A difference means the interpreter and the code
   generator disagree about the mapping, and that alone is the bug.

3. **Single-step the SSA.** Turn on tracing and read the operation stream:

   ```python
   from ninetoothed.interpret import Tracer

   result = interpret(
       arrangement,
       application,
       inputs=(x, out),
       trace=Tracer(),  # everything
   )
   print(result.render_trace(limit=60))
   ```

   Narrow it down with `Tracer(opcodes={"mem.store"})` or
   `Tracer(program_ids=[2])` to look at one program instance, and use
   `Tracer(watch={"%6"})` to dump the full contents of one value.

4. **Stop at a specific operation.** `Tracer(breakpoints={...})` maps an opcode
   or an SSA location to a callback invoked before the operation runs; raising
   from it aborts with a `TraceStop` carrying the event.

---

## 6. Known limitations

Keep these in mind before concluding that a mismatch is a real bug:

- `bfloat16` and `float8` are refused, not approximated. Cast to `float32` to
  interpret a kernel that uses them.
- `mem.atomic_add` is not implemented: accumulation order is not deterministic,
  so there is no well-defined reference. Kernels that use it (for example the
  `test_data_ptr.py` reduction) cannot be interpreted as-is.
- `math.rand` is not implemented.
- Target intrinsics such as `triton.cdiv` arrive as `call.*` and have no CPU
  implementation; replace them with an equivalent arithmetic expression.
- Reading a view before every outer dtype level has been indexed is rejected.
  For `x.tile((1, B)).tile((1, -1))`, index down with `x[0, i]` first.
- The frontend lowers an application by reading its source, so the arrangement and
  the application must be module-level functions; a `lambda` cannot be lowered.
  (`interpret` says so explicitly when it is handed one.)
- Docstrings are fine. A bare literal statement is a no-op in Python, so the
  frontend skips it instead of emitting an `arith.constant` string, and a
  documented arrangement or application lowers exactly like an undocumented one.
- A `block_size()` meta parameter cannot be pinned from the call site. The
  compiled kernel accepts meta overrides only under the symbol's auto-generated
  internal name (`BLOCK_SIZE_0`, `BLOCK_SIZE_1`, …; the counter is global), so
  `kernel(x, out, BLOCK_SIZE=16)` raises `Unknown kernel arguments: BLOCK_SIZE`.
  The interpreter still takes the friendly name via `symbols={"BLOCK_SIZE": 16}`,
  so the two sides can silently disagree on the block width. Since the width
  decides how a row is tiled, that changes the answer, not just the performance.
  Use a plain constant in the arrangement when both sides must agree, as
  `cross_validate.py` does.
- Execution is single-threaded, one program instance at a time. Use small inputs.

Run `python -c "from ninetoothed.interpret import format_support_matrix; print(format_support_matrix())"`
on the server to see the exact operation set of the checkout you are testing.

---

## 7. Quick reference

```shell
# CPU only: the interpreter's own suite
python -m pytest tests/test_interpret.py -v

# CPU only: everything in the repo that does not need torch (248 tests)
python -m pytest -q tests/test_backend_registry.py tests/test_compiler_cache_runtime.py \
    tests/test_compiler_entrypoints.py tests/test_emitter_boundaries.py \
    tests/test_eval.py tests/test_getitem.py tests/test_interpret.py \
    tests/test_ir_immutability.py tests/test_kernel_ir.py \
    tests/test_layout_transfer_analysis.py tests/test_lowering_inference.py \
    tests/test_materializer_registry.py tests/test_naming.py \
    tests/test_ssa_application_lowering.py tests/test_ssa_first_backend_lowering.py \
    tests/test_ssa_pass_pipeline.py tests/test_ssa_validation.py \
    tests/test_target_profiles.py tests/test_unsqueeze.py

# With a GPU
python -m pytest tests/ -q
python cross_validate.py

# Lint and format, matching the project's ruff configuration
ruff check src tests
ruff format --check src tests
```

---

## 8. Requirements coverage

A checklist of what each acceptance criterion maps to, so it can be re-verified
instead of taken on trust.

### Pass criteria

| Criterion | Where it is satisfied | How to check |
| --- | --- | --- |
| 1. Existing tests still pass, project style followed, design doc / support matrix / usage docs present | `src/ninetoothed/interpret/`; `docs/source/python_api/interpret.rst`; `README.md`; this file | `pytest tests/ -q` on a GPU host; `ruff check src tests`; `ruff format --check src tests` |
| 2. No CUDA execution path imported or called; tests run where CUDA is invisible | `ninetoothed.language` resolves `libdevice` lazily; nothing on the interpreter path imports Triton | `test_importing_the_interpreter_pulls_in_no_cuda_runtime` runs a bare `python -c` and asserts no `triton`/`torch`/`tilelang` module is loaded |
| 3. Five application classes covered: elementwise, broadcast, masked load/store on non-divisible sizes, row reduction, branch/loop | `tests/test_interpret.py` semantics group | `pytest tests/test_interpret.py -k "elementwise or broadcast or masked or reduction or loop" -v` |
| 4. `float32` / `int32` / `bool`; integer and boolean exact, float compared at `rtol=1e-3, atol=1e-3` | `interpret/dtypes.py`; `DEFAULT_RTOL` / `DEFAULT_ATOL` in `interpret/diff.py` | `test_integer_arithmetic_is_bit_exact`, `test_bool_comparison_and_select`, `test_elementwise_does_not_widen_float32` |
| 5. At least three programs agree before vs. after the default optimisation passes; agreement with the GPU backend on A100 | `test_default_pipeline_is_semantics_preserving` (3 programs) and `test_matmul_survives_the_whole_default_pipeline`; GPU agreement via `cross_validate.py` | `pytest tests/test_interpret.py -k default_pipeline -v`; then `python cross_validate.py` on the server |
| 6. Trace reproduces and localises intermediates; unsupported op / dtype / access fail loudly, never fall back to GPU or skip silently | `interpret/trace.py`; `interpret/registry.py`; `interpret/errors.py` | `test_trace_records_opcodes_masks_and_program_ids`, `test_unsupported_dtype_is_rejected_with_context`, `test_target_intrinsics_are_reported_clearly`, `test_unmasked_out_of_bounds_store_is_reported` |

### Excellence criteria

| Criterion | Where it is satisfied | How to check |
| --- | --- | --- |
| 1. `dot` / matmul, softmax and richer dtypes, dynamic shapes and access patterns | `interpret/operations/reduce.py` (`linalg.dot`), `tensorops.py` (nested tiles, `expand`, `squeeze`); int32/bool/float32; symbolic shapes | `test_matmul_with_expand_and_squeeze`, `test_matmul_with_masked_k_tail`, `test_softmax_uses_other_for_masked_lanes`, `test_transpose` |
| 2. Filter traces by program id or SSA operation; single stepping, breakpoints, watch values | `Tracer(program_ids=..., opcodes=..., watch=..., breakpoints=..., on_event=...)` | `test_trace_records_opcodes_masks_and_program_ids`, `test_access_map_can_be_restricted_to_one_instance`; see §5 step 3–4 |
| 3. Compare every pass against its predecessor and name the first pass and SSA operation that changes semantics | `compare_passes` → `PipelineDiff`, `PassStage`, `PipelineDiff.localize()` | `test_compare_passes_names_the_first_diverging_pass`, `test_localization_names_the_instance_and_the_store` |
| 4. On a differential failure, export the SSA, input data, shapes, dtypes and random seed | `build_reproduction` → `Reproduction`; `random_inputs`; `interpret(..., seed=...)` | `test_reproduction_carries_ssa_data_shape_dtype_and_seed`; `reproduction.to_json()` |
| 5. Interface and operation implementations well decoupled; new operations added by local extension and reused by frontend and backend tests | `@register("opcode", category=..., summary=...)` in `interpret/operations/`; `OperationSpec`; `support_matrix()` | `test_support_matrix_declares_known_gaps`, `test_every_frontend_opcode_is_covered_or_declared` |
| 6. Merged into the NineToothed main branch | outside the scope of this checkout | open a PR against `InfiniTensor/ninetoothed` |

### Adding an operation

The registry is the extension point, and nothing in the interpreter core needs to
change. The most common case is a target intrinsic: ``ntl.<name>`` lowers to a
``call.<name>`` opcode, and the interpreter refuses those by default with a
message naming the intrinsic. Registering a handler for that opcode teaches the
interpreter about it:

```python
import numpy as np

from ninetoothed.interpret import format_support_matrix, register
from ninetoothed.interpret.operations.common import bind_data, materialize, operands


@register("call.logaddexp", category="call", summary="log(exp(a) + exp(b)).")
def _handle_logaddexp(state, operation):
    left, right = operands(state, operation)
    bind_data(
        state,
        operation,
        np.logaddexp(
            materialize(left, state.context), materialize(right, state.context)
        ),
    )


assert "call.logaddexp" in format_support_matrix()
```

`format_support_matrix()` picks the new entry up immediately, so an application
that uses `ntl.logaddexp` now interprets instead of raising, and
`test_every_frontend_opcode_is_covered_or_declared` keeps the declared set honest
when the frontend grows.

A registered handler always wins over the built-in `call.*` refusal, so an
intrinsic can be adopted one opcode at a time.
