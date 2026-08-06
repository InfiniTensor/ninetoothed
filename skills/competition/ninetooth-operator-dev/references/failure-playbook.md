# Failure Playbook

Use this file when tests, builds, generated source, or benchmarks fail.
Keep it as a diagnosis path, not a log dump. Capture the first useful signal,
classify the failure, make the smallest justified fix, and rerun the same
command before broadening verification.

## Minimum Repro Loop

1. Reproduce with the narrowest command that still reaches the failure.
2. Classify the failure into one layer below before editing code.
3. Inspect the closest upstream test or source anchor for that layer.
4. Apply the smallest fix that explains the observed failure.
5. Rerun the same command first, then the full task verification command.

Do not hide a failure by removing required tests, widening tolerances without
evidence, silently excluding required layouts, or replacing a blocked benchmark
with a performance claim.

## Failure Record

- command:
- observed output:
- first failing assertion:
- suspected layer:
- files inspected:
- root cause:
- minimal fix:
- rerun command:
- rerun result:
- unsupported scope:
- stop reason:

## Common Failure Layers

### Requirement Extraction

Symptoms:

- expected output is unclear;
- shape, dtype, broadcast, layout, or boundary rules are missing;
- the test oracle disagrees with the task statement;
- the requested case is outside visible repository support.

Inspect:

- `operator-task-contract.md`;
- the closest upstream operator test named in `repo-map.md`;
- the task prompt or self-test `TASK.md`.

Minimal fix path:

- fill unknown task-contract fields before implementation;
- add an explicit assumption only when a nearby upstream pattern supports it;
- record unsupported dtype, layout, device, benchmark, or AOT scope instead of
  silently dropping it.

### Arrangement

Symptoms:

- `simulate_arrangement` target tensors differ from the intended tile or
  broadcast shape;
- output shape is plausible but lanes contain values from the wrong source
  positions;
- tail blocks contain real input values where `other=` sentinels or masks should
  appear;
- `tile`, `expand`, `pad`, `permute`, `flatten`, `ravel`, `squeeze`, or
  `unsqueeze` order diverges from the closest upstream test.

Inspect:

- `${NINETOOTHED_REPO}/tests/test_debugging.py`;
- `${NINETOOTHED_REPO}/src/ninetoothed/debugging.py`;
- the nearest upstream arrangement test for the operator family.

Minimal fix path:

- run or adapt `simulate_arrangement(arrangement, tensors)` on a small shape;
- compare source and target index tensors before debugging application math;
- fix one meta-operation at a time and rerun arrangement simulation;
- preserve explicit `other=` or shape options when tail lanes are expected.

### Application

Symptoms:

- arranged tensors look correct but `torch.allclose` fails;
- reduction values are numerically unstable;
- scalar or constexpr parameters are ignored or treated as tensors;
- output assignment uses the wrong temporary or leaves the store path unchanged.

Inspect:

- the matching upstream `application` function, such as add, pow, softmax,
  matmul, addmm, attention, or max_pool2d;
- `verification-matrix.md` for oracle and tolerance expectations.

Minimal fix path:

- reduce the failing input to a small hand-checkable case;
- compare with the PyTorch or repository oracle named in the task contract;
- fix math, dtype cast, accumulator, mask, or output assignment directly;
- tune tolerances only after formula and dtype policy match the contract.

### Tensor Metadata

Symptoms:

- generated code uses an unexpected shape, stride, dtype, or constexpr
  parameter;
- a meta-operation changes tensor rank but dtype metadata is not squeezed or
  expanded with it;
- non-contiguous, offset, or jagged tensors behave like contiguous dense
  tensors;
- source and arranged tensors disagree about `shape`, `dtype`, `other`,
  `shape_options`, `value`, or `constexpr`.

Inspect:

- `${NINETOOTHED_REPO}/src/ninetoothed/tensor.py`;
- upstream generation, attention, matmul, conv2d, eval, or jagged tests that
  adjust dtype metadata after shape operations.

Minimal fix path:

- trace metadata through arrangement before editing application code;
- update dtype shape metadata only where the arrangement changes rank;
- confirm stride and offset behavior with a layout-sensitive test or mark the
  layout unsupported if the task allows that boundary;
- keep scalar `Tensor(0, constexpr=True, value=...)` behavior distinct from
  runtime tensor inputs.

### Correctness Test

Symptoms:

- failure occurs only in one dtype, boundary shape, broadcast case, or layout
  variant;
- the test compares the wrong output tensor or uses a stale expected value;
- tolerance is too strict for the stated dtype, or too loose without rationale;
- a CUDA-only helper is run on a non-CUDA device.

Inspect:

- `verification-matrix.md`;
- `${NINETOOTHED_REPO}/tests/utils.py` for device fixture behavior;
- the nearest upstream test's `torch.allclose`, `torch.equal`, `rtol`, and
  `atol` choices.

Minimal fix path:

- confirm the reference expression and input construction first;
- keep ordinary, boundary, dtype, broadcast, and layout cases separate enough to
  identify the failing requirement;
- change tolerance only when dtype and upstream patterns justify it;
- rerun the exact failing test before running the wider suite.

### Generated Source

Symptoms:

- `kernel._source` or generated cache files are missing expected launch code;
- generated source contains wrong masks, redundant loads or stores, missing
  stride terms, or unexpected auto-tuning wrappers;
- source generation fails with unsupported caller, invalid bounds, or missing
  symbols.

Inspect:

- `${NINETOOTHED_REPO}/src/ninetoothed/generation.py`;
- `${NINETOOTHED_REPO}/tests/test_generation.py`;
- `performance-diagnostics.md` for generated-source evidence fields.

Minimal fix path:

- capture the generated source path from `kernel._source` or
  `ninetoothed.generation.CACHE_DIR`;
- inspect the exact function or `launch_<kernel_name>` symbol, not only the
  Python wrapper;
- map suspicious generated expressions back to arrangement metadata or
  application math;
- if auto-tuning generation fails, check symbol bounds before changing
  benchmark code.

### AOT Build

Symptoms:

- `.cpp`, `.h`, or `.so` artifacts are missing;
- `launch_<kernel_name>` cannot be found in the generated header or loaded
  library;
- `python -m triton.tools.compile` or `nvcc -shared -arch native` fails;
- runtime launch fails for a specific divisibility, contiguity, size, or stride
  variant.

Inspect:

- `${NINETOOTHED_REPO}/src/ninetoothed/aot.py`;
- `${NINETOOTHED_REPO}/tests/test_aot.py`;
- generated files under the requested `output_dir`.

Minimal fix path:

- verify `kernel_name`, `caller`, `output_dir`, generated header signature, and
  exported `launch_<kernel_name>` symbol;
- decide whether the failure is in source generation, C/C++ compilation, shared
  library loading, or runtime launch;
- compare tensor shape and stride metadata with dispatcher divisibility and
  contiguity checks;
- rerun the smallest AOT case before rerunning large operator tests.

### Benchmark

Symptoms:

- timing is missing, unstable, or incomparable with the baseline;
- benchmark uses different shape, dtype, layout, device, warmup, repetition, or
  cache state for baseline and candidate;
- auto-tuning picks a failing or unexpectedly slow configuration;
- `_KernelLaunchError` is converted to infinite timing by auto-tuning.

Inspect:

- `${NINETOOTHED_REPO}/src/ninetoothed/build.py`;
- `${NINETOOTHED_REPO}/src/ninetoothed/auto_tuner.py`;
- `${NINETOOTHED_REPO}/tests/test_auto_tuner.py`;
- `performance-diagnostics.md` for the benchmark contract.

Minimal fix path:

- record baseline, candidate, input sizes, dtype, layout, device, warmup,
  repetitions, and cache state before interpreting timing;
- rerun with one small correctness-friendly size and one performance-relevant
  size when hardware allows;
- separate kernel launch failure from slow-but-correct performance;
- if the environment blocks timing, record the blocker and static inspection
  evidence instead of estimating speed.

### Integration

Symptoms:

- the operator works in isolation but import, export, example, or package-level
  tests fail;
- test names, fixture style, output path, or benchmark entry point diverges from
  nearby files;
- the skill framework lint fails after adding examples, scripts, or references.

Inspect:

- `repo-map.md` for file placement;
- `script-index.md` for helper command intent;
- `scripts/lint_skill_structure.py` and `tests/test_structure.py` for framework
  checks.

Minimal fix path:

- align naming and file placement with the closest upstream or skill-framework
  pattern;
- fix missing references, headings, or paths directly;
- keep generated artifacts and local competition source files out of the skill
  package.

### Environment

Symptoms:

- CUDA-only tests are skipped or fail because CUDA is unavailable;
- Torch, Triton, compiler, or device runtime is missing;
- local cache or output directory state affects reproducibility;
- a command fails before reaching the intended test assertion.

Inspect:

- the exact command output up to the first environment blocker;
- upstream device selection helpers and benchmark/build references;
- `git status --short --branch` to separate local edits from generated files.

Minimal fix path:

- record the blocker, not a guessed result;
- narrow the command until it either reaches the intended assertion or proves
  the environment is unavailable;
- keep cache state, device count, and build output paths in the failure record;
- use static generated-source or AOT inspection only as a named fallback.

## Stop Conditions

Stop and report instead of guessing when:

- the required dtype or layout is unsupported by the repository;
- the task needs compiler-core changes outside the stated scope;
- the benchmark environment is unavailable;
- the failure cannot be reproduced.
- the first failing signal is an environment blocker rather than operator logic;
- the only possible fix would remove a required check, fake a result, or bypass
  build, generated-source, AOT, or benchmark failure;
- two targeted reruns after a minimal fix still point to different unrelated
  layers.

When stopping, return the failure record, inspected files, smallest attempted
fix, rerun command, residual risk, and the next concrete file or command another
agent should inspect.
