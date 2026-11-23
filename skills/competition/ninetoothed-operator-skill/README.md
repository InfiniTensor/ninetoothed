# NineToothed Operator Development Skill

A reusable `.skill` that turns a general AI coding agent into a reliable NineToothed
operator developer. It does not try to be clever; it enforces a **low-variance,
correctness-first workflow** so an agent completes unfamiliar operator tasks in one
pass: classify → copy an existing operator → extract semantics → check layout →
implement minimally → verify with pytest → benchmark if relevant → self-correct via
a failure table → run a compliance gate.

## Why this design

The evaluation runs a fixed agent on 8 hidden operator tasks, once each, with no
retry on skill-induced failure. So the skill optimizes for **not failing**:

- **Repo-grounding** — force the agent to read and mimic a working, tested operator
  in the repo instead of writing a niche DSL from memory.
- **Executable verification loop** — scripts generate repo-style tests, run the full
  CI check sequence, and benchmark, so "tests pass" is nearly automatic.
- **NineToothed failure-recovery table** — symptom → root cause → minimal fix, so the
  agent self-corrects within its single run.

## Package structure

- `SKILL.md` — the main workflow for the AI agent (trigger, 4-lane dispatch,
  verification loop, failure table, compliance gate).
- `references/`
  - `operator-dev-map.md` — condensed, source-verified NineToothed API and patterns.
  - `task-to-example-index.md` — task type → which existing operator to copy.
  - `failure-recovery.md` — symptom → root cause → minimal fix.
- `scripts/`
  - `run_ci_checks.sh` — ruff format + ruff check + style checker + pytest.
  - `gen_test_scaffold.py` — generate a repo-style pytest scaffold.
  - `bench.py` — `triton.testing.do_bench` wrapper reporting the five required facts.
- `examples/` — four self-test task records (also the with-skill evidence).
- `tests/` — how to validate the skill itself.
- `conftest.py` — puts the repo root on `sys.path` so examples run from the root.
- `HONOR_CODE.md`, `REFERENCE.md` — signed integrity and disclosure.

## How to use

1. Open the NineToothed repository in an AI coding tool.
2. Have the agent read `SKILL.md` before implementing any operator task.
3. The agent follows the fixed workflow; it consults `references/` on demand and
   runs `scripts/` to verify.
4. Record correctness, benchmark, and any failure diagnosis.

## Scope

- Elementwise / broadcast operators.
- Reduction / block operators.
- Layout-sensitive operators (non-contiguous, stride, offset).
- Performance, benchmarking, generated-source, AOT build, and failing-test diagnosis.

## Out of scope

No compiler-core changes (`src/ninetoothed/` is never modified), no hidden-task
answers, no hard-coded evaluation names, no API keys or online-only dependencies.

## Self-test tasks

| ID | Lane | Operator | Correctness | Benchmark |
|---|---|---|---|---|
| T1 | Elementwise / broadcast | elementwise multiply | 1 passed | via T4 |
| T2 | Reduction / block | row softmax | 1 passed | reusable |
| T3 | Layout-sensitive | 2-D multiply, contiguous + non-contiguous | 2 passed | — |
| T4 | Performance / diagnosis | multiply benchmark + real diagnosis loop | see record | yes |

All correctness tests pass on real hardware (WSL2, RTX 5060, CUDA 13.0,
torch 2.13.0+cu130, triton 3.7.1). Run everything with:

```bash
pytest skills/competition/ninetoothed-operator-skill/examples -v -p no:cacheprovider
```
