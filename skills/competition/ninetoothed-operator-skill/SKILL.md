---
name: ninetoothed-operator-skill
description: >-
  Guides an AI coding agent through developing, testing, debugging, benchmarking,
  and integrating operators (kernels) in the NineToothed repository. Trigger this
  skill whenever a task involves writing or modifying a NineToothed operator,
  adding or fixing its pytest, aligning it with a PyTorch reference, diagnosing a
  failing test, inspecting generated source, configuring an ahead-of-time (AOT)
  build with `ninetoothed.build`, or benchmarking a kernel. Also trigger when the
  task mentions NineToothed concepts such as `arrangement`, `application`, `tile`,
  `expand`, `Tensor`, `Symbol`, `ninetoothed.make`, `ntl.`, or the arrange-and-apply
  paradigm. Do NOT trigger for tasks unrelated to NineToothed operator work, and
  never modify the compiler core under `src/ninetoothed/`.
---

# NineToothed Operator Development Skill

You are developing an operator in the **NineToothed** repository, a Triton-based
DSL that uses the **arrange-and-apply** paradigm. Your goal is a correct, tested,
repo-style-compliant implementation whose performance does not obviously regress.

**Core principle: favor low variance over cleverness.** Each evaluation task runs
once and is not retried on skill-induced failure, so follow the fixed workflow
below in order, do not skip steps, and self-correct using the failure table when
anything breaks.

## The one rule that matters most

Correctness and tests are the bulk of the score. **Do not optimize or refactor
before the operator is correct and its test passes.** Proceed in the fixed order:
understand → find reference → implement minimally → verify → (only then) performance.

---

## Step 0 — Classify the task (pick exactly one lane)

Read the task and route it to ONE of four lanes. This decides which existing
operator you copy from and which pitfalls apply.

| Lane | Task looks like | Copy from | Details |
|---|---|---|---|
| **L1 Elementwise / broadcast** | add, relu, gelu, masked add, per-element math, broadcasting | `tests/test_add.py` | broadcast via `expand`, boundary via `other=` |
| **L2 Reduction / block** | softmax, sum, max, mean, layernorm, pooling, block statistics | `tests/test_softmax.py`, `tests/test_max_pool2d.py` | reduce with `ntl.max/sum`, fill `other=-inf`/`0` |
| **L3 Layout-sensitive** | transpose input, sliced/strided/offset input, non-contiguous | `tests/test_getitem.py`, `tests/test_matmul.py` | never assume contiguous; use `permute`, check strides |
| **L4 Performance / diagnosis / integration** | benchmark, generated source, AOT build, failing test, perf regression | `tests/test_matmul.py`, `docs/source/build.rst` | see `references/operator-dev-map.md` §4 |

If a task spans lanes (e.g. a layout-sensitive reduction), follow the primary lane
and apply the extra checks from the secondary lane.

See `references/task-to-example-index.md` for the full mapping and what to look
for in each reference file.

---

## Step 1 — Read the repository BEFORE writing code

Do not write anything yet. First read, in this order:

1. The reference operator for your lane (from the table above). Copy its structure.
2. `README.md` and `docs/source/basics.rst` if you are unsure of the paradigm.
3. `references/operator-dev-map.md` — the condensed NineToothed API and patterns.
4. Any existing operator with a similar shape signature.

**Why:** NineToothed is a niche DSL. Mimicking a working, tested operator in the
repo is far more reliable than writing from memory.

## Step 2 — Extract the operator semantics

Write down (in the task record) before implementing:

- **Inputs / outputs**: how many tensors, which is the output.
- **Shape constraints**: ranks, which dims must match, reduction axis.
- **Dtype constraints**: fp16/fp32/bf16/int; does the accumulator need fp32?
- **Broadcasting**: do any inputs broadcast against each other?
- **Boundary / masking**: what fills out-of-bounds elements? (reductions need
  `other=float("-inf")` for max, `0` for sum).

## Step 3 — Decide the layout (do NOT skip)

Explicitly answer: **is every input guaranteed contiguous?**

- If the task mentions transpose, slicing, stride, offset, `permute`, `.T`, or
  "non-contiguous" → it is an **L3** concern even if the primary lane differs.
- You MUST design at least one **non-contiguous input test** for layout-sensitive
  work (e.g. pass `x.T`, `x[::2]`, or `x.narrow(...)`), and compare against a
  contiguous baseline.

This step alone protects the two layout-sensitive hidden tasks. Most agents fail
them by silently assuming contiguous inputs.

## Step 4 — Implement (minimal, arrange-and-apply)

Follow the pattern from your reference operator. Key NineToothed rules:

- Two equivalent forms: the `@ninetoothed.jit` decorator (simple ops, tile in the
  annotation) or `make(arrangement, application, tensors)` (complex ops). Match
  whichever your reference uses.
- **application parameters are blocks (the second-outermost tensor), not whole
  tensors.**
- Block sizes: `Symbol("BLOCK_SIZE", meta=True)` to let the compiler auto-tune;
  `constexpr=True` when a fixed value is needed (e.g. softmax row width), and pass
  it at call time.
- **Multi-input ops: the outermost shapes of all arranged tensors MUST match.**
  Align them with `expand`; drop size-1 dims with `.dtype.squeeze(...)`.
- Keep the patch minimal: no unrelated refactors, no mass reformatting, do not
  touch `src/ninetoothed/`.

## Step 5 — Verify (correctness closes the loop)

1. Write a **PyTorch reference** (`expected = torch.<op>(...)`).
2. Generate a repo-style pytest: `python scripts/gen_test_scaffold.py` (parametrizes
   `device` from `get_available_devices()`, seeds via `conftest.py`, asserts
   `torch.allclose`, `atol` for fp16).
3. Run `bash scripts/run_ci_checks.sh` (ruff format, ruff check, style checker, pytest).
4. **No GPU?** `get_available_devices()` returns empty and tests SKIP — that is
   normal. Record the skip honestly; NEVER rewrite skip/failure as pass.

## Step 6 — Performance (only after correctness, only if relevant)

For performance-sensitive tasks (L4, or when asked):

- Benchmark with `python scripts/bench.py` (wraps `triton.testing.do_bench`).
  Report all five: **baseline, input sizes, command, result, conclusion.**
- Baseline = PyTorch reference, or an existing repo implementation, or no-skill output.
- To inspect codegen: pass `kernel_name=` and `output_dir=` to `make`, or use
  `ninetoothed.debugging.simulate_arrangement`. For AOT: `ninetoothed.build` with
  `meta_parameters` for auto-tuning (see `references/operator-dev-map.md` §4 T4).

## Step 7 — On failure, consult the recovery table

If anything errors, DO NOT thrash. Open `references/failure-recovery.md`, match the
symptom, apply the minimal fix, re-run. Record: symptom → root cause → minimal fix
→ re-run command → re-run result.

## Step 8 — Compliance gate (a single violation invalidates the submission)

Before declaring done, confirm ALL of:

- [ ] Changes are confined to `skills/competition/<skill-name>/` (never `src/ninetoothed/`).
- [ ] No hard-coded hidden task names or evaluation answers.
- [ ] No deleted, weakened, or unconditionally skipped tests.
- [ ] No API keys, credentials, or online-only dependencies.
- [ ] No fabricated results; skips and failures reported honestly.
- [ ] Patch is minimal and matches repo style (naming, blank lines, imperative commits).

---

## Output

When done, produce a task record following the template in
`examples/<lane>/README.md`: task description, agent execution summary, added/modified
files, correctness command + result, benchmark (if any), and failure diagnosis (if any).

