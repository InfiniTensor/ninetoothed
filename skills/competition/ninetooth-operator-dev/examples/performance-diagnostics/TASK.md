# Self-Test Task: Performance / Diagnostics / Integration

## Input Task

Create a NineToothed diagnostic self-test named `aot_add_autotune_diagnostics`
in a temporary upstream test file such as
`tests/test_aot_add_autotune_diagnostics.py`.

Diagnostic target:

- build an AOT add-style kernel through `ninetoothed.build`;
- include at least two meta-parameter candidates so auto-tuning and generated
  dispatch are exercised;
- verify the built kernel against `torch.add(input, other, alpha=alpha)`;
- inspect generated artifacts under a dedicated output directory;
- record benchmark or blocker evidence for candidate and baseline;
- diagnose any failure by first useful signal, not by guessing.

Required input sizes:

- correctness-friendly size: `1127`;
- performance-relevant size: `20260128` or the largest locally safe fallback;
- dtype coverage: `torch.float32` required, `torch.float16` optional with
  explicit tolerance;
- layout: contiguous rank-1 tensors only.

Required generated-source/AOT evidence:

- generated `.cpp`, `.h`, and `.so` artifact presence or blocker;
- `launch_<kernel_name>` symbol or header evidence;
- auto-tuning cache `.csv` and `.fingerprint` state when `ninetoothed.build`
  creates them;
- whether `application_with_auto_tuning` or generated dispatch appears where
  expected.

Unsupported scope for this self-test: non-contiguous tensors, multi-output
kernels, compiler-core changes, hidden benchmark answers, and speedup claims
without a completed benchmark.

## Agent Execution Summary

Status: task specification only. No upstream patch, correctness run, benchmark
run, AOT build, or failure diagnosis has been recorded for this example yet.

Execution steps for the agent using this task:

1. Read the repository anchors below before writing the diagnostic test.
2. Add the smallest upstream self-test patch, preferably only
   `tests/test_aot_add_autotune_diagnostics.py`.
3. Reuse the add/auto-tuning `premake`, arrangement, and application style from
   `tests/test_aot_auto_tuning.py` instead of inventing a new operator family.
4. Write all generated files into a dedicated subdirectory under
   `ninetoothed.generation.CACHE_DIR`; clean it before the run and after
   successful completion.
5. Run the repro/correctness command first. If it fails, stop at the first
   useful failure signal and fill the failure diagnosis fields.
6. Run benchmark evidence only after correctness and artifact checks complete,
   or record a concrete blocker.

## Repository Files Inspected

- `${NINETOOTHED_REPO}/tests/test_aot_auto_tuning.py`: add-style
  `premake`, config set, `ninetoothed.build`, output directory cleanup, and
  correctness comparison against `torch.add`.
- `${NINETOOTHED_REPO}/tests/test_aot.py`: AOT build patterns, generated
  `kernel_name`, `caller`, `output_dir`, dtype-specific tensors, correctness
  tests, and static non-power-of-two coverage.
- `${NINETOOTHED_REPO}/tests/test_generation.py`: generated source
  inspection via `kernel._source` and `application_with_auto_tuning` checks.
- `${NINETOOTHED_REPO}/src/ninetoothed/build.py`: auto-tuned build
  dispatch, `.csv` cache, `.fingerprint`, generated source/header mutation, and
  lazy kernel paths.
- `${NINETOOTHED_REPO}/src/ninetoothed/auto_tuner.py`: timing cache
  behavior and `triton.testing.do_bench` use.
- `${NINETOOTHED_REPO}/tests/utils.py`: device selection helper via
  `get_available_devices`.

## Patch Summary

Expected patch surface:

- add `tests/test_aot_add_autotune_diagnostics.py` in the upstream NineToothed
  checkout;
- do not edit NineToothed compiler internals unless the diagnostic proves a
  focused bug and the task explicitly asks for a fix;
- do not commit generated `.cpp`, `.h`, `.so`, `.csv`, `.fingerprint`, cache
  directories, benchmark artifacts, or raw logs;
- do not change existing AOT or auto-tuning tests except to mirror their local
  patterns in the new diagnostic test.

Expected diagnostic test shape:

- define `arrangement(input, other, alpha, output, block_size=None)`;
- define `application(input, other, alpha, output)` as
  `output = input + alpha * other`;
- define `premake(size=None, dtype=None, block_size=None)` using rank-1 input,
  rank-1 other, scalar alpha, and rank-1 output tensors;
- call `ninetoothed.build(..., meta_parameters=("block_size",), kernel_name=...,
  output_dir=...)` with at least two `block_size` candidates;
- verify output against `torch.add(input, other, alpha=alpha)`;
- inspect generated artifacts and summarize the result without pasting long
  generated files.

## Correctness Command

```bash
cd ${NINETOOTHED_REPO}
python -m pytest tests/test_aot_add_autotune_diagnostics.py -q
```

## Correctness Result

Not run yet. Replace this line only with real pytest output or a concise summary
after the command above has been executed.

Required result fields after execution:

- command:
- environment:
- generated artifact summary:
- cases passed:
- failures:
- fix and rerun, if any:

## Benchmark Command

Run only after the correctness command passes and generated artifacts are
accounted for.

```bash
cd ${NINETOOTHED_REPO}
python - <<'PY'
import shutil

import torch
import triton.testing

import ninetoothed
import ninetoothed.generation
from tests.test_aot_add_autotune_diagnostics import build_add_kernel
from tests.utils import get_available_devices

devices = get_available_devices()
if not devices:
    raise SystemExit("blocked: no available device")

device = devices[0]
dtype = torch.float32
size = 20260128
output_dir = ninetoothed.generation.CACHE_DIR / "selftest_aot_add_bench"

shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(parents=True, exist_ok=True)

kernel = build_add_kernel(
    size=size,
    dtype=ninetoothed.float32,
    device=device,
    output_dir=output_dir,
    kernel_name="selftest_add",
)

input = torch.randn((size,), dtype=dtype, device=device)
other = torch.randn((size,), dtype=dtype, device=device)
alpha = torch.randn((), dtype=torch.float64)
output = torch.empty_like(input)

candidate = lambda: kernel(input, other, alpha, output, size, ninetoothed.float32)
baseline = lambda: torch.add(input, other, alpha=alpha)

print(
    {
        "size": size,
        "dtype": str(dtype),
        "device": str(device),
        "output_dir": str(output_dir),
        "candidate_ms": triton.testing.do_bench(candidate),
        "baseline_ms": triton.testing.do_bench(baseline),
        "artifacts": sorted(path.name for path in output_dir.iterdir()),
    }
)
PY
```

## Benchmark Result

Not run yet. Record the real benchmark output, hardware/runtime blocker, or
generated-source/AOT fallback evidence here. A blocked benchmark is acceptable
only if it names the attempted command, device/runtime blocker, intended
baseline, input sizes, output directory, and artifacts inspected.

## Performance Conclusion

Pending real benchmark, generated-source, or AOT evidence. Do not claim a
speedup, slowdown, parity, or regression until the benchmark command has run
successfully or a concrete blocker plus fallback artifact evidence has been
recorded.

## Failure Diagnosis

Not run yet.

Required failure record if the repro, build, artifact inspection, or benchmark
fails:

- command:
- observed output:
- first failing assertion:
- suspected layer: generated source, AOT build, benchmark, integration, or
  environment.
- files inspected:
- root cause or blocker:
- minimal fix:
- rerun command:
- rerun result:
- unsupported scope:
- stop reason:

## Minimal Fix or Workaround

Not run yet. Fill this section only after a concrete failure or blocker exists.

Acceptable minimal fixes or workarounds include:

- correct a stale or reused output directory in the diagnostic test;
- remove generated artifacts before a run and rerun the same command;
- adjust the diagnostic config set only when the failing signal is a config
  mismatch;
- record an environment blocker when CUDA, Triton, `nvcc`, or benchmark runtime
  is unavailable;
- use generated-source/AOT artifact inspection as fallback evidence only when
  benchmark timing is blocked.

## Risks and Unsupported Scope

- This task checks generated-source, AOT, auto-tuning, and benchmark evidence for
  one add-style kernel; it does not claim coverage of all operator families.
- Non-contiguous tensors, multi-output kernels, alternate callers, multi-device
  launch, and compiler-core changes are out of scope unless a later task asks
  for them.
- Auto-tuning and benchmark results depend on cache state, device, Triton
  version, and warmup behavior; record these details before drawing conclusions.
- Generated artifacts must remain local cache output, not committed project
  files.
- If the first failure is an environment blocker, stop and record the blocker
  instead of inventing a timing or root cause.
