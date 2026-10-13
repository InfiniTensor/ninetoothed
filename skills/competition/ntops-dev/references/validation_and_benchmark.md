# Validation and Benchmark Evidence

Use this reference when recording correctness, benchmark, or diagnosis output.

Source: commands follow the checked-in `ntops` pytest layout and the bundled benchmark scripts. Recorded numbers must come from actual command output; this document does not define official competition thresholds.

## Correctness

Run the narrowest relevant pytest first:

```bash
cd ntops
python3 -m pytest tests/test_<op>.py -q
```

Record:

- command,
- environment if known,
- pass/fail status,
- failing shape/dtype/device if any,
- tolerance policy,
- fix and rerun result.

## Benchmark

Benchmark requires a single CUDA GPU for meaningful `ntops` operator evidence.

Record:

- GPU model,
- CUDA, PyTorch, Triton, and Python versions,
- input shapes,
- dtype,
- baseline implementation,
- warmup count,
- measured iterations,
- latency or throughput,
- conclusion.

Minimum benchmark shape:

```text
operator=<op>
baseline=torch reference or existing implementation
device=cuda:0
dtype=float16 and/or float32
warmup>=10
iters>=50
```

## Generated Source Inspection

When benchmark shows a material gap, run:

```bash
cd <submission-root>
python3 skills/competition/ntops-dev/scripts/inspect_generated_source.py \
  softmax --shape 1024,1024 --dtype float16
```

For an operator that the tool cannot trigger generically, compile it with the task's
real inputs first, then inspect the exact emitted file or the latest isolated cache:

```bash
python3 skills/competition/ntops-dev/scripts/inspect_generated_source.py \
  <op> --source "$HOME/.ninetoothed/<generated>.py"
python3 skills/competition/ntops-dev/scripts/inspect_generated_source.py \
  <op> --no-trigger --latest 3
```

Then record:

- generated source path,
- line count and file size,
- `tl.*` or `triton.language.*` operation counts, especially load, store, dot, exp, erf, where, arange, sum, and max,
- notable `float32`/`float16` casts,
- whether autotune or heuristics appear,
- likely bottleneck and next experiment.

Treat this as a diagnosis aid, not final proof. A performance claim still needs a follow-up benchmark.

## Promotion Protocol

For an optimization candidate, keep one immutable baseline and change one factor at a
time. Use identical inputs, explicit fixed block/tile configuration,
`max_num_configs=1`, warmup, iterations, and measurement method. Run at least three
rounds in interleaved baseline/candidate order and compare medians. Reject and revert
the candidate when correctness fails, measured improvement is below 5%, or the result
is unstable. Preserve the rejected result as diagnostic evidence rather than presenting
it as an optimization success.

## Pending CUDA Evidence

If local CUDA is unavailable, do not invent results. Record:

- exact command to run on the remote RTX 4090 D/CUDA environment,
- expected output artifact,
- current status: `pending CUDA run`.

## Failure Diagnosis

Use this structure:

```text
Symptom:
Command:
Input shape/dtype/device:
Observed:
Expected:
Likely root cause:
Minimal fix:
Verification:
Remaining risk:
```
