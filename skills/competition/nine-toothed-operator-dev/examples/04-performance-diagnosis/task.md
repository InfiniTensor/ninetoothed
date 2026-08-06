# Self-Test 04: Performance Regression or Build Diagnosis

## Category

Performance / diagnosis / integration.

## Input Task Statement

Investigate a performance regression, generated source issue, AOT build problem, or failing test in a NineToothed operator. Produce a minimal fix or a clear mitigation with reproducible validation.

Required coverage:

- exact failing or slow command
- first relevant error line or benchmark evidence
- root cause hypothesis
- minimal fix or mitigation
- rerun command and result
- benchmark conclusion

## Candidate Scenarios

- Softmax benchmark slower after a tile/block change.
- Generated source contains redundant loads or unexpected stores.
- AOT build fails because of a stale or missing configuration.
- A failing test is caused by wrong reduction identity fill.

## Expected Agent Workflow

1. Reproduce the failure or benchmark result.
2. Classify the failing layer: task, arrangement, application, wrapper, test, build, or benchmark.
3. Inspect generated source or AOT logs if relevant.
4. Apply the smallest fix.
5. Rerun correctness.
6. Rerun benchmark or document the blocker.

## Failure Record

Symptom:

```text
AOT smoke test failed in the remote RTX 4090 environment. After fixing PATH for `python`, the remaining blocker was a missing `nvcc` executable.
```

Command:

```shell
/usr/local/miniconda3/envs/py312/bin/python -m pytest tests/test_aot.py::test_add[False-45327-dtype0-bf16-cuda] -q
```

Evidence:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'nvcc'
```

Root cause:

```text
The PyTorch CUDA runtime and driver were available, but the AOT path invokes the CUDA compiler `nvcc`. The rented image exposes CUDA 12.8 runtime and PyTorch 2.9.1+cu128, but `nvcc` was not on PATH.
```

Fix or mitigation:

```text
Install or expose CUDA Toolkit `nvcc`, or choose a cloud image that includes the full CUDA toolkit rather than runtime-only PyTorch. Keep `PATH=/usr/local/cuda/bin:$PATH` and verify with `nvcc --version` before rerunning AOT tests.
```

## Correctness Verification

Command:

```shell
/usr/local/miniconda3/envs/py312/bin/python -m pytest tests/test_add.py -q
/usr/local/miniconda3/envs/py312/bin/python -m pytest tests/test_softmax.py -q
```

Result:

```text
tests/test_add.py: 1 passed in 8.69s
tests/test_softmax.py: 1 passed in 1.86s
```

## Benchmark Verification

Command:

```shell
/usr/local/miniconda3/envs/py312/bin/python /root/ninetoothed-skill-work/selftest_custom.py
```

Result:

```text
add ninetoothed_ms 0.05119999870657921, torch_ms 0.020479999482631683
softmax ninetoothed_ms 0.05432000011205673, torch_ms 0.021503999829292297
```

## Re-Verification (closed loop, 2026-07-02)

The mitigation was applied: a new session used an image with the full CUDA Toolkit (`nvcc` release 12.8, V12.8.61) exposed at `/usr/local/cuda/bin`.

Command:

```shell
nvcc --version   # Build cuda_12.8.r12.8/compiler.35404655_0
python -m pytest tests/test_aot.py -q
```

Result:

```text
.s..........                                                             [100%]
11 passed, 1 skipped in 892.77s (0:14:52)
```

The full AOT test suite passes once `nvcc` is available, confirming the root-cause judgment: the failure was a toolchain/provisioning issue, not an operator or compiler-code defect. Note the long wall time (~15 min for 12 tests) — AOT tests invoke `nvcc` per configuration, which matters when budgeting evaluation time.

## Conclusion

Correctness and lightweight benchmark evidence were collected. The AOT blocker was diagnosed (missing `nvcc`), the mitigation was specified (full CUDA Toolkit image + PATH), and re-verification on a compliant image shows `tests/test_aot.py` fully passing. This is a complete diagnose → mitigate → re-verify loop across two environments.
