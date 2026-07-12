# SELFTEST-EW-001: Elementwise And Broadcast-Like Coverage

## Task Description

Inspect ntops `add` and `relu`, add focused dtype/shape/boundary coverage, run
available correctness checks, and prepare a correctness-gated short benchmark.
The case tests whether guidance keeps elementwise work narrow and evidence
honest when local CUDA is unavailable.

## Agent Execution Summary

Both no-skill and skill-guided sessions changed only `tests/test_add.py` and
`tests/test_relu.py`. The baseline produced the smaller structural patch. The
skill-guided session produced broader requirement, failure, and evidence
records and more runtime-oriented cases. Neither local session could establish
CUDA correctness because its Windows environment lacked a usable runtime.

Separate preserved server evidence later established selected public
correctness and the included add timing record.

## Production And Test Files Changed

- Production files: none.
- Baseline tests: `tests/test_add.py`, `tests/test_relu.py`.
- Skill-guided tests: `tests/test_add.py`, `tests/test_relu.py`.
- Patch character: test-only; no compiler-core or wrapper change.

## Correctness

Recorded focused command from the ntops repository:

```bash
python -m pytest tests/test_add.py tests/test_relu.py
```

Server result excerpt: [server_correctness_excerpt.log](server_correctness_excerpt.log).

- add: `8 passed`, return code `0`.
- relu: `16 passed`, return code `0`.
- Final independent combined add/relu/softmax rerun: `32 passed`
  ([log](../../../reports/independent_audit/focused_correctness_recheck.log)).
- These selected public checks do not prove every broadcast or dtype class.

## Benchmark

Raw record: [benchmark.csv](benchmark.csv); generated command summary:
[benchmark.md](benchmark.md).

Portable replay command with the recorded parameters:

```bash
python scripts/run_ntops_microbenchmark.py --task-id SELFTEST-EW-001 --operator add --shape 1024x1024 --dtype float32 --warmup 10 --repeat 30 --baseline pytorch --candidate ntops --output-csv benchmark.csv --output-md benchmark.md
```

| Baseline | Candidate | Shape | DType | Layout | Warmup | Repeat | Median result |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| PyTorch | ntops | `1024x1024` | float32 | contiguous | 10 | 30 | 0.017408 ms vs 0.066000 ms |

Correctness guard: `PASS`. The baseline/candidate median ratio is `0.263758`;
ntops took about 3.79 times the PyTorch latency for this selected input. This
is a real selected-shape regression, not a broad performance conclusion.

## Failure And Closure

- Symptom: local pytest import stopped before CUDA execution.
- Root cause: no usable local Triton/CUDA runtime for the repository stack.
- Minimal response: preserve static checks and defer runtime claims; do not
  alter production code or weaken tests.
- Re-run result: selected server checks and benchmark completed successfully.

## No-Skill Versus Skill

- Baseline advantage: smaller patch with direct structural rank checks.
- Skill advantage: clearer requirement extraction, evidence status, failure
  record, and runtime-oriented test intent.
- Neutral conclusion: process quality improved; the fresh-session comparison
  alone did not prove a correctness or performance advantage.

## Unsupported Or Unverified

- Full PyTorch-style broadcasting is not claimed.
- Other dtypes, shapes, layouts, generated source, AOT, and hidden evaluation
  remain unverified.
