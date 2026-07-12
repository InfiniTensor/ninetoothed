# Self-test tasks

Four tasks, one per hidden-task family (rules §4.2.2 / §4.3). Each is run twice:
**no-skill** (baseline agent, no `.skill` loaded) and **with-skill** (this
`.skill` loaded), same model / harness / repo / budget. Score each run with
`verifier_spec.md`.

| # | Task | Family | Input range | Must include |
|---|------|--------|-------------|--------------|
| 1 | masked elementwise add | elementwise / broadcast | `(B,M,N)` + `(1,N)` broadcast + bool mask; fp16 & fp32 | broadcast shape reasoning; `other=` fill; non-pow2 shape |
| 2 | parameterized mean reduction | reduction / blocking | reduce last dim; `reduction in {none, mean, sum}`; fp16 & fp32 | fp32 accumulate; NaN/Inf boundary; **benchmark** |
| 3 | pixel_unshuffle (space-to-depth) | layout-sensitive | `(B,C,H,W)`, factor r∈{2,3,4}; contiguous + non-contiguous | layout via ravel/flatten/permute OR contiguous fast-path; non-contiguous test |
| 4 | softmax perf-regression diagnosis | perf / diagnosis | a deliberately inefficient softmax + a benchmark.csv | generated-source inspection; **benchmark** before/after; minimal fix |

At least tasks 2 and 4 include a benchmark (rules §4.2.4).

## Per-task record (fill for each run)

For every task and every condition (no-skill / with-skill) record:

1. Input task statement (verbatim).
2. Agent execution summary (key steps, tool calls).
3. Produced artifact: kernel code + wrapper + test (or fix patch) — summary.
4. Correctness command + result (paste `run_correctness_matrix.py` summary).
5. Benchmark (tasks 2, 4): baseline, input sizes, command, GB/s or TFLOPS,
   compute/memory-bound conclusion.
6. If a failure was diagnosed: symptom → root cause → minimal fix → verification.

## A/B comparison table (per task)

| metric | no-skill | with-skill | delta |
|--------|----------|------------|-------|
| Pass@1 (matrix pass rate) | | | |
| RubricScore (0–10, §4.4) | | | |
| produced a benchmark? | | | |
| tool calls / wall time | | | |
| negative transfer (broke a passing case)? | | | |

Target: with-skill ≥ no-skill on Pass@1 and RubricScore for all 4 tasks; no
negative transfer.
