# SELFTEST-RED-001: Reduction And Blocking

## Task Description

Inspect softmax and nearby normalization patterns, add deterministic axis,
dtype, shape, and stability coverage, and run a short benchmark only after
focused CUDA correctness.

## Agent Execution Summary

Both sessions limited their changes to `tests/test_softmax.py`. The baseline
created broader stability and float64-output coverage, but its saved patch did
not apply in the isolated server workspace. The skill-guided patch was smaller,
applied there, produced `12 passed`, and unlocked the recorded short benchmark.

## Production And Test Files Changed

- Production files: none.
- Baseline test: `tests/test_softmax.py`.
- Skill-guided test: `tests/test_softmax.py`.
- Patch character: test-only reduction semantics and boundary coverage.

## Correctness

Focused command from the ntops repository:

```bash
python -m pytest tests/test_softmax.py
```

Evidence: [server_correctness_excerpt.log](server_correctness_excerpt.log).

- Existing public server check: `8 passed`.
- Skill-guided isolated check: `12 passed in 8.22s`.
- Final independent combined add/relu/softmax rerun: `32 passed`
  ([log](../../../reports/independent_audit/focused_correctness_recheck.log)).
- Nearby reference evidence recorded `96 passed` for layer norm and `64 passed`
  for RMS norm; these are context, not changes made by this patch.

## Benchmark

Raw record: [benchmark.csv](benchmark.csv); generated command summary:
[benchmark.md](benchmark.md).

Portable replay command with the recorded parameters:

```bash
python scripts/run_ntops_microbenchmark.py --task-id SELFTEST-RED-001 --operator softmax --shape 64x1024 --dtype float32 --warmup 10 --repeat 30 --baseline pytorch --candidate ntops --output-csv benchmark.csv --output-md benchmark.md
```

| Baseline | Candidate | Shape | DType | Layout | Warmup | Repeat | Median result |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| PyTorch | ntops | `64x1024` | float32 | contiguous | 10 | 30 | 0.023568 ms vs 0.071728 ms |

Correctness guard: `PASS`. The baseline/candidate median ratio is `0.328575`;
ntops took about 3.04 times the PyTorch latency for this selected input. This
is a real selected-shape regression, not a broad performance conclusion.

## Failure And Closure

- Symptom: baseline server `git apply --check` failed.
- Root cause boundary: saved patch portability, not softmax runtime correctness.
- Minimal response: do not run baseline CUDA tests from a patch that did not
  apply; preserve the failure and evaluate the applicable skill-guided patch.
- Re-run result: skill-guided patch applied, focused correctness passed, then
  the short benchmark ran.

## No-Skill Versus Skill

- Baseline advantage: broader explicit stability, axis, dtype, and shape cases.
- Skill advantage: smaller applicable patch, stronger evidence packaging,
  successful server correctness, and a correctness-gated benchmark.
- Conclusion: the skill-guided arm completed the executable path, while the
  baseline retained richer test ideas that were not server-evaluated.

## Unsupported Or Unverified

- No long benchmark, generated-source, AOT, dispatch, or broad dtype conclusion.
- The recorded timing covers only one contiguous float32 shape.
