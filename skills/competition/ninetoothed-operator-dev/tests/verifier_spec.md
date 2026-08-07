# Verifier spec

How each self-test run is scored. Mirrors the official single-task rubric
(rules §4.4): 0–10 per task across six sub-scores. Used identically for
no-skill and with-skill runs so the delta is meaningful.

## Sub-scores (per task, total 10)

| Sub-score | Max | Pass condition | How to check |
|-----------|-----|----------------|--------------|
| Task completion | 4 | operator semantics / shape / dtype / boundary or diagnosis goal met | `run_correctness_matrix.py` full pass = 4; partial = 1–3; none = 0 |
| Test & verification | 2 | specified tests pass and the loop is closed | all matrix cases pass + commands+results recorded = 2; partial = 1; not run/failed = 0 |
| Performance awareness | 1 | benchmark / generated-source analysis / sound optimization rationale present | benchmark.csv + Roofline verdict, or `inspect_generated_source` evidence |
| Patch minimality | 1 | no unrelated refactor, no mass reformat, no breaking change | `git diff` review |
| Repo-style consistency | 1 | naming / structure / error handling / test & doc style match repo | ruff/black clean + matches the NineToothed repo's own `tests/test_*.py` style |
| Process & compliance | 1 | reproducible trace, no secrets, no network, no test bypass, no hidden answers | inspect the run log |

## Deterministic gates (auto-checkable)

- **Correctness**: `python scripts/run_correctness_matrix.py <test_file>` exits 0
  (no FAILED/ERROR; SKIPPED allowed only when CUDA absent).
- **Benchmark present** (tasks 2, 4): a CSV with ≥3 input sizes and a stated
  compute/memory-bound conclusion exists.
- **Generated-source evidence** (task 4): `inspect_generated_source.py` output
  captured for the before and after kernels.
- **Semantic contract** (reduction/scatter-class tasks):
  `scripts/inspect_generated_source.py --contract <contract>` exits 0 — the
  generated source contains the primitive the claimed semantics requires
  (fold op for reductions, `tl.atomic_*` for many-to-one writes, exp+max for
  stable softmax). Choose `<contract>` by op:
  `reduction`/`stable_softmax`/`atomic`/`matmul`/`elementwise` (there is no
  `layout` or `perf-diag` contract — those families map onto these). A
  violation (exit 2) is treated as a correctness failure even when the pytest
  matrix is green: matrices sample inputs, the contract reads what was
  compiled.
- **Compliance**: no `eval(`/`exec(`/network calls in produced scripts; no test
  deleted vs the provided scaffold.

### Three-gate mapping (KernelBench/MusaCoder convention)

Kernel-generation benchmarks such as KernelBench (`ModelNew` interface,
`torch.utils.cpp_extension.load_inline`, nvcc) and MusaCoder (MooreEval sandbox,
nvcc/MUSA toolchain) score a submission on three gates: ① compiles, ② is
numerically correct, ③ is legal (no banned `aten::*` fallback). This skill's
task interface differs (`wrapper.py:solve()` + `kernel.py`, not a `ModelNew`
class; NVIDIA-only, no `mcc`/MUSA backend), so the three gates map onto
existing mechanisms rather than being literal ports:

| # | Gate | Where it lives | Notes |
|---|------|-----------------|-------|
| ① | Compiles | companion evaluation harness repo's `proxy_runner/oracle.py`'s `from wrapper import solve` (entry-point-parse contract) + `compiled`/`compile_attempts` fields in `proxy_runner/run_episode.py` (subprocess timeout 900s) and `proxy_runner/reward.py` (`R_NOCOMPILE = -2.0`) | Not a separately reported gate in the rubric — a compile/import failure folds into `completion=0` via the oracle test's ERROR outcome. No distinct pass/fail line item exists yet; see gap below. |
| ② | Numerically correct | companion harness's `proxy_runner/oracle.py` (`assert got.shape == ref.shape`) + this skill's own `scripts/run_correctness_matrix.py` MERE/MARE metric (mean/max relative error, per-dtype thresholds) | Stricter than a naive `torch.allclose(atol=rtol=1e-2)` — MERE/MARE thresholds are 1–2 orders of magnitude tighter per dtype (see the table above) and are checked across the task's random-input dtype matrix. No migration needed. |
| ③ | Legal (no banned `aten::*` fallback) | this skill's own `evaluation/skill_eval/reward_hacking_guard.py::banned_fallback_analysis()` — AST scan for `matmul`/`mm`/`bmm`/`conv*d`/`sum`/`mean`/`softmax`/`layer_norm`/... on `kernel.py`/`kernel_fixed.py`/`wrapper.py`, excluding the `ntl.*`/`ninetoothed.*` in-kernel DSL namespace. Wired into the companion harness's `proxy_runner/rubric_scorer.py`: caps `completion` to 1 (`_legality_violation`) and zeroes `compliance` (`score_compliance`) on a hit. | This gate did not exist before — it is a real gap that has now been closed. See "Known gap, closed" below. |

**Known gap, closed**: prior to this addition, a solution could satisfy `②`
by calling `torch.sum`/`torch.matmul`/etc. directly in `wrapper.py` instead of
driving the NineToothed kernel, and nothing in the rubric would catch it —
`_uses_ninetoothed()` only checked for the *presence* of `ninetoothed`/`make(`
anywhere in the file, which a one-line stub import would satisfy. The new
`banned_fallback_analysis()` closes this by scanning for the actual banned
call, not just the absence of a NineToothed import.

**Declared fallback vs deception at gate ③**: `banned_fallback_analysis()`
distinguishes two cases. A torch.* fallback in a file carrying a *valid*
`# ninetoothed-fallback: <feature> — <reason>` marker (SKILL.md Stop Rules,
"Declared fallback — never silent") is reported with a `declared_fallback:`
prefix; without the marker — or with a `<feature>` not in the recognised
inexpressible set — the prefix is `banned_fallback:`. Both keep the completion
cap (no NineToothed kernel was delivered), so there is never a *higher* score
to be won by declaring.

The `declared_fallback:` prefix is a **route-to-review signal, not an automatic
compliance pass.** The static scan can verify the marker names a real
inexpressible feature; it cannot verify that feature actually applies to *this*
task (a lazy agent could stamp a valid feature name onto an expressible op).
Recommended scoring: `banned_fallback:` zeroes `compliance` automatically;
`declared_fallback:` is surfaced for a human/second-model check of "is this
task genuinely inexpressible?" — compliance is zeroed only if that check finds
the declaration inapplicable. This keeps the honest-disclosure path open
without letting one comment line auto-exempt the compliance sub-score.

**Still open**: gate ① has no dedicated pass/fail signal distinct from a full
correctness run — a compile failure and a correctness failure both currently
surface as `completion=0` with a `completion_error` string. Splitting these
(e.g. a `compiled: bool` field on `RubricResult`) is future work, not required
for the award threshold below. Likewise, the companion harness's
`rubric_scorer.py` does not yet consume the `declared_fallback:` prefix — it
currently treats all gate-③ hits alike (the *safe* default: it never
auto-exempts compliance). Wiring the route-to-review handling above is
follow-up work there.

## Award-threshold mapping (rules §4.5)

The skill is "on track" when, across the 4 self-tests:

- with-skill total ≥ no-skill total on every task, and
- ≥ 2 tasks demonstrate a valid performance verification (tasks 2 & 4), and
- 0 negative-transfer regressions (no case that passed no-skill fails
  with-skill).

These mirror the hidden-task thresholds: ≥48/80 pre-scaling, ≥5/8 tasks with
completion ≥3/4, ≥2 tasks with valid performance work.
