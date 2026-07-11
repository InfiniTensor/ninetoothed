---
name: ninetoothed-op-dev-skill
description: NineToothed operator development skill for arrangement/application implementation, correctness tests, benchmark, generated source, AOT build, non-contiguous/stride/offset handling, performance regression diagnosis, and failing test repair. Use when implementing, testing, benchmarking, debugging, repairing, or documenting NineToothed operators, Triton kernels, or InfiniTensor ninetoothed repositories.
---

# NineToothed Operator Development Skill

## When to use

- Implement / fix a **NineToothed** kernel (`arrangement` / `application` / `ninetoothed.make`)
- Correctness vs PyTorch or repo tests; layout (stride/offset); benchmark / AOT / failing-test repair

## Do not use for

- Compiler-core edits (`src/ninetoothed/generation.py`, `cudaifier.py`, …) unless the task explicitly requires it
- Fabricating benchmarks, deleting/skipping tests, or guessing hidden evaluation answers

## Hard compliance

1. No hidden answers, hardcoded hidden task names, API keys, or anti-eval logic.
2. Minimal patches only; log every command for offline reproduction.

---

## Execution decision tree (MANDATORY)

**Stop conditions:** Do not write kernel code until D1–D3 are done. Skip a branch only if the task card marks it N/A.

### D1 — Emit task card first (before any code)

Output a short card with **all** of:

| Field | Must state |
|-------|------------|
| Math | operator semantics (exact formula / PyTorch op) |
| Shape | every input/output shape (symbolic OK) |
| dtype | per tensor |
| Broadcast | rules or `N/A` |
| Layout | contiguous / stride / offset / view requirements |
| Boundaries | empty, mask, padding, fp16 tol, etc. |
| Reference | PyTorch op **or** nearest repo test/kernel path |

If any field is unknown → search the repo (D3) or mark `TODO: verify`; do not invent.

### D2 — Route by family → nearest pattern

Pick **one** primary family, then open the listed starting points (under `--repo-root`):

| Family | Route to |
|--------|----------|
| Elementwise / broadcast | `tests/test_add.py`, `tests/test_expand.py`, `tests/test_pow.py`; examples `ops/.../add.py` |
| Reduction / block | `tests/test_softmax.py`, `tests/test_max_pool2d.py`, `tests/test_matmul.py` |
| Stride / offset / layout | `tests/test_clone.py`, `tests/test_data_ptr.py`, `tests/test_getitem.py` |
| Perf / diagnosis / AOT | `tests/test_generation.py`, `tests/test_aot.py`, `tests/test_auto_tuner.py` |

Read `references/03`–`05` / `07`–`09` only for the chosen family. **Reuse** that file’s `arrangement` / `application` / test style — do not invent APIs.

### D3 — `rg` before code (non-negotiable)

From the NineToothed repo root, search then read hits **before** editing:

```bash
rg -n "arrangement|application|ninetoothed.make" tests/ -g "*.py" | head
rg -n "<op_keyword>" tests/ src/ninetoothed/ -g "*.py" | head
```

Record: nearest test path + nearest kernel/example path. No `rg` → no patch.

### D4 — Implement minimal patch

- Touch only files the task needs; mirror `tests/test_<op>.py` layout.
- Prefer `ninetoothed.make` / `@ninetoothed.jit` patterns already in-repo.
- fp16 defaults: `atol=2e-2, rtol=1e-2` unless the task tightens.

### D5 — Layout branch (if layout ≠ contiguous-only)

**Forced:**

1. Build inputs as non-contiguous views (transpose / `as_strided` / slice) matching the card.
2. In tests: `assert not inp.is_contiguous()` (and on other strided tensors the card requires).
3. **Forbidden:** `.contiguous()` / materializing copies unless the task card explicitly allows it.
4. Prefer stride/offset load patterns from `test_clone.py` / `references/05_*`.

### D6 — Correctness gate

```bash
pytest <test_file> -v --tb=short
```

Log command + exit code under `logs/correctness/`. **FAIL → D7. PASS + perf task → D8. Else → D9.**

### D7 — Failure loop (forced format)

On every failure, output exactly:

1. **Symptom** — command + error excerpt (≤20 lines)
2. **Hypothesis** — one minimal, falsifiable claim (shape / dtype / stride / wrong op / codegen)
3. **Minimal fix** — smallest diff that tests the hypothesis
4. **Re-verify** — **same original command**; new exit code + excerpt

Rules:

- One hypothesis per iteration; do not shotgun-edit.
- If the message says “broadcast/layout” but values look like wrong op (`*` vs `+`), read `application` **before** changing `arrangement`.
- Loop or declare **unsupported** with reason — never claim PASS on FAIL.

### D8 — Performance branch (only if task is perf-sensitive)

**Forced order — skip any step → no performance claim:**

1. **Correctness first** — D6 PASS at every config you will time (e.g. each `BLOCK_SIZE`).
2. **Fixed inputs** — document exact shapes, dtype, device; do not change mid-bench.
3. **Warmup** — ≥5 iterations (per config) before timing.
4. **Repeat** — ≥20 timed iters (or project standard); report **median / p10 / p90** (and `spread_ratio=p90/median`); note if CI unavailable.
5. **Fair baseline** — same dtype/device/size; disable or disclose autotuning (`meta=True`); constexpr `BLOCK_SIZE` when comparing tiles.
6. **Conclusion bounds** — one line: what is comparable vs **N/A / incomparable** (e.g. autotuning window). Never invent numbers.

Log under `logs/benchmark/`.

### D9 — Report (mandatory fields)

- `changed_files` — path + one-line purpose
- `commands` — copy-pasteable, execution order
- `correctness_result` — PASS/FAIL + log path
- `benchmark_result` — numbers **or** `N/A` + reason
- `failure_diagnosis` — D7 block **or** `N/A`
- `unsupported_cases` — explicit list (may be empty)

---

## Compact execution spine

Use this only as a checklist; **decisions live in D1–D9**.

| # | Action | Gate |
|---|--------|------|
| 1 | Task card (D1) | all fields filled |
| 2 | Family route (D2) + `rg` (D3) | nearest paths recorded |
| 3 | Minimal patch (D4) + layout rules (D5) | no illegal `.contiguous()` |
| 4 | Correctness (D6) | PASS or enter D7 |
| 5 | Perf if required (D8) | correctness-first + fair baseline |
| 6 | Report (D9) | all mandatory fields |

Arrange-and-apply sketch (verify against repo, do not invent):

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

kernel = ninetoothed.make(arrangement, application, tensors)
```

---

## References (on demand)

| File | When |
|------|------|
| `00_repo_map.md` | repo layout |
| `01_ninetoothed_concepts.md` | arrangement / application |
| `03_elementwise_broadcast_patterns.md` | elementwise / broadcast |
| `04_reduction_block_patterns.md` | reduce / block |
| `05_layout_stride_offset_patterns.md` | stride / offset |
| `06_correctness_testing_patterns.md` | pytest patterns |
| `07_benchmark_patterns.md` | D8 details |
| `08_generated_source_aot_debugging.md` | codegen / AOT |
| `09_failure_diagnosis_playbook.md` | D7 details |
| `10_patch_minimality_checklist.md` | before submit |
| `11_unsupported_cases.md` | limits |

Fallback: `src/ninetoothed/` + `tests/` under `--repo-root`.

## Helpers

- `env_check.py --repo-root .` — env report only
- `make_task_card.py` — task card from prompt (no `--repo-root`)
- `run_correctness.py` / `run_benchmark.py` — logged wrappers (`--repo-root`)
- `repo_pattern_index.py` / `check_patch_minimality.py` — repo scan / diff (`--repo-root`)
- `score_task.py` / `gate_eval.py` — evidence scoring (no `--repo-root`)
- `summarize_run.py` / `quick_validate.py` — text / package checks (no `--repo-root`)

Scripts must not bypass tests or fabricate results.

## Rubric focus (0–10)

| Pts | Optimize for |
|-----|----------------|
| 4 | Semantics, shapes, dtypes, boundaries |
| 2 | Tests + verification trail |
| 1 | Perf awareness when required (D8) |
| 1 | Minimal diff |
| 1 | Repo style |
| 1 | Compliance + reproducible logs |

**Search the repo; never guess hidden task names or answers.**
