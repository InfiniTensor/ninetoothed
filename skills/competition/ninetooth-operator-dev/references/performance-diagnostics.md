# Performance Diagnostics

Use this reference only for performance-sensitive tasks, generated-source
inspection, AOT build diagnosis, or benchmark-required tasks from
`verification-matrix.md`.

If the task is not performance-sensitive, do not load this file.

## When to Use

Load this file when the task touches any of these upstream anchors:

- `src/ninetoothed/auto_tuner.py`
- `src/ninetoothed/build.py`
- `src/ninetoothed/generation.py`
- `tests/test_generation.py`
- `tests/test_aot.py`
- `tests/test_aot_auto_tuning.py`

These files show how NineToothed generates source, chooses configs, writes
cache artifacts, and validates AOT output.

## Benchmark Contract

Record benchmark evidence in a way another agent can rerun without guessing.

| Field | Record |
| --- | --- |
| task id | The operator or diagnosis task being measured. |
| baseline | The reference implementation or prior version. |
| candidate | The new implementation, config, or patch under review. |
| input sizes | At least one correctness-friendly size and one performance-relevant size. |
| dtype | Exact dtype(s) used for each run. |
| device | Exact device type and, when relevant, count or index. |
| layout | Contiguous, strided, sliced, transposed, or offset. |
| warmup | Warmup count or the reason warmup was not run. |
| repetitions | Repetition count or the repo default used. |
| command | Exact command or harness invocation. |
| result | Timing summary, unit used, and whether the result was stable. |
| conclusion | Faster, slower, tied, blocked, or inconclusive, plus why. |
| blocker | Hardware, build, dependency, or environment issue if the run did not finish. |

Benchmark rules:

- compare like with like: same device, dtype, layout, and input shape;
- do not claim a speedup from a single run unless the repo's benchmark helper
  only emits one stable measurement;
- note whether caches were cold or warm, especially for auto-tuned kernels;
- for auto-tuning tasks, include the config set and whether the cache file or
  CSV was reused.

## Generated Source Review

Use this checklist when the task asks for generated source inspection or when a
performance regression may come from code generation rather than math.

Check for:

- redundant loads;
- redundant stores;
- repeated broadcast computation;
- unnecessary dtype conversion;
- missing stride/contiguous information;
- excessive boundary masks;
- suspicious tile/block configuration.

Also check the generated artifacts that NineToothed writes under
`ninetoothed.generation.CACHE_DIR`:

- `.py` source files generated from the DSL;
- `.cpp` and `.h` files for AOT builds;
- `.so` libraries when AOT compilation succeeds;
- `.csv` auto-tuning caches;
- `.fingerprint` files used to detect stale builds.

For generated source reviews, capture the exact symbol or file name you
inspected, such as `launch_<kernel_name>` or `application_with_auto_tuning`.

## AOT Build Review

Use this checklist when the task involves `ninetoothed.build`, an AOT failure,
or a generated kernel that should produce a reusable library.

Check for:

- generated files present:
- expected symbols:
- launch configuration:
- build command:
- failure output:
- minimal fix:

Look for these concrete signs in the upstream flow:

- source and header files are emitted before the launch function is loaded;
- the exported symbol matches `launch_<kernel_name>`;
- the build directory contains the generated `.cpp`, `.h`, and `.so` artifacts;
- auto-tuning builds may also emit `.csv` and `.fingerprint` files;
- the failure output names the missing file, missing symbol, compiler error, or
  config mismatch rather than just saying "build failed".

## Regression Policy

If performance is worse and cannot be fixed immediately, the agent must record
the evidence, likely root cause, affected shapes, and fallback recommendation.

Never turn a blocked or inconclusive benchmark into a performance claim.

Required regression record:

- baseline and candidate command;
- input sizes and dtype;
- layout or contiguity state;
- warmup and repetition counts;
- whether the cache was cold, warm, or reused;
- generated source or AOT evidence if relevant;
- likely root cause: layout handling, tile/block choice, launch config, dtype
  conversion, boundary mask, or cache behavior;
- fallback recommendation: smallest acceptable fix, safe workaround, or explicit
  unsupported scope.

If the environment blocks the run, record the blocker, the intended baseline,
the intended candidate, and any static inspection result. Do not guess the
timing.

## Result Template

- task id:
- baseline:
- candidate:
- input sizes:
- dtype:
- device:
- layout:
- warmup:
- repetitions:
- command:
- result:
- generated source or AOT evidence:
- conclusion:
- likely root cause:
- fix or workaround:
- blocker:
- unsupported scope:
