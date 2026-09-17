---
name: ninetoothed-operator-dev
description: >-
  Write, validate, optimize, and debug NineToothed DSL operators. Use when the
  task involves authoring or modifying a NineToothed kernel (arrangement +
  application), adding a correctness test against PyTorch, inspecting generated
  Triton source, configuring AOT build, designing a benchmark, or diagnosing a
  failing test / performance regression in a repo that depends on the
  `ninetoothed` package. Covers elementwise/broadcast, reduction/blocking,
  layout-sensitive (non-contiguous/stride/offset), and performance/diagnosis
  operator families.
license: See REFERENCE.md
metadata:
  target_package: "ninetoothed>=0.25.0"
  cross_agent: true
  permissions:
    file-read: true     # reads ~/.ninetoothed cache, repo source, reference files
    file-write: true    # writes test files, benchmark CSVs, AOT build outputs
    network: false      # no network calls
    shell: true         # runs pytest subprocess and aot_build_smoke.sh (list-form argv only)
---

# NineToothed Operator Development

NineToothed is a Triton-based DSL. You write two Python functions — an
**arrangement** (compile-time tiling/permute/expand of symbolic `Tensor`s) and
an **application** (the per-tile compute) — and call
`ninetoothed.make(arrangement, application, tensors)` to get a kernel handle.
This skill routes you to the right family playbook and runnable scripts.

## 1. Trigger

Activate when ANY of these hold:

- Writing or editing a file that imports `ninetoothed` / `ninetoothed.language`,
  or lives under `src/ninetoothed/`, `ops/ninetoothed/`, or an `ntops` tree.
- The user asks to implement / test / benchmark / optimize / debug a NineToothed
  operator, or to inspect its generated source or AOT build.
- A NineToothed kernel fails: compile error, `allclose` mismatch, AOT build
  failure, or benchmark regression.

Do **not** activate (avoid negative transfer):

- Pure PyTorch refactor with no NineToothed kernel involved.
- An equivalent op already exists in the repo / `ntops` — reuse it instead.
- Docs / CI / build-config changes unrelated to a kernel.
- Editing the NineToothed compiler internals (out of scope; see §5).

## 2. Workflow

Follow these steps; load the linked reference only when you reach that step.

| Step | What to do | Load |
|------|------------|------|
| 1. Spec | Extract inputs/outputs, shape, dtype, broadcast, stride/offset, boundary | — |
| 2. Classify | Pick the operator family AND screen expressibility — if the spec needs a feature the DSL cannot express, take the declared-fallback path (§3) instead of writing a kernel | `references/operator-taxonomy.md` |
| 3. Find prior art | Search repo for a similar arrangement before writing new | `scripts/find_similar_ops.py` |
| 4. Arrange + apply | Write `arrangement` + `application` in repo style | `references/<family>.md` |
| 5. Debug arrangement | Run `debug_arrangement` to confirm no OOB and correct tile mapping — before kernel compile | `scripts/debug_arrangement.py` |
| 6. Oracle + correctness | Generate a PyTorch reference, run shape×dtype×layout matrix | `scripts/gen_pytorch_oracle.py`, `scripts/run_correctness_matrix.py` |
| 7. Inspect generated source | Read tile / num_warps / num_stages from the cached Triton source; for reduction/scatter-class ops also check the semantic contract (`--contract`) — a green matrix alone does not prove the semantics | `scripts/inspect_generated_source.py` |
| 8. Benchmark + Roofline | Compare vs PyTorch baseline, classify compute/memory-bound | `scripts/bench_compare.py`, `references/perf-diag.md` |
| 9. Diagnose failures | Map symptom → root cause → minimal fix → re-verify | `references/common-errors.md` |

Families: `elementwise` · `reduction` · `layout` · `perf-diag`
(see `references/operator-taxonomy.md` for the decision rule).

## 3. Stop Rules

- **Success**: correctness passes on the full matrix AND, for perf-sensitive
  ops, you have a benchmark with a stated compute/memory-bound conclusion.
- **Numerical-only**: if `allclose` fails *only* by tolerance on fp16/bf16,
  first apply the dtype-upcast fix (`references/common-errors.md`), then re-test
  once. Do not loosen tolerance to force a pass.
- **Classify before retrying**: after any failure, classify it before the next
  iteration. Two repair targets, different actions:

  | Failure type | Evidence | Action |
  |---|---|---|
  | `code_error` | matches `common-errors.md`; OOB from `debug_arrangement`; first occurrence | Fix `kernel.py` / `wrapper.py`. |
  | `guidance_error` | same symptom ≥ 2 times; API mismatch not in `common-errors.md` | The skill text has a gap. Note it in the trace — do **not** keep fixing the kernel. Report as "skill gap: <description>" and fall through to PyTorch. |
  | `dsl_limit` | task needs a feature in the known-inexpressible set (see below), or an error text reaching for a missing atomic primitive | Terminal — no repair target exists on either side. Stop iterating and take the declared-fallback path below. |
  | `unknown` | neither rule fires | Escalate: add the pattern to `common-errors.md` after root-cause analysis. |

  Use `scripts/failure_classifier.py`
  (`classify(error_text, oob_count, symptom_history, task_features=)`) for
  deterministic classification; `screen_task_features()` runs the same
  expressibility check pre-attempt at workflow step 2. If `guidance_error`
  fires, the fix belongs in `references/<family>.md`, not in the kernel —
  this is a distinct repair path.

- **Declared fallback — never silent**: some tasks are outside the DSL's
  expressible set in `ninetoothed` 0.25.0 — data-dependent indexing,
  many-to-one scatter (needs atomic RMW), value-dependent control flow,
  dynamic output shape, cross-tile communication (definitions:
  `failure_classifier.KNOWN_INEXPRESSIBLE`). Do not burn iterations on these
  and do not quietly re-implement the op some other way. Instead:
  1. State the limitation in the report: which feature, why it is
     inexpressible, and what was checked.
  2. Implement the PyTorch fallback in the wrapper, and mark it with a
     comment line `# ninetoothed-fallback: <feature> — <one-line reason>` so
     the legality gate can tell a declared fallback from a disguised one.
  A declared fallback scores low on completion but is not a compliance
  violation; a silent one is both.
- **Give up to fallback**: after 3 `code_error` attempts on the same root
  cause, stop and report a PyTorch fallback with the diagnosis (same
  declaration marker as above).
- **Hard cap**: 10 total iterations on one operator.

## 4. Artifact Checklist

Every completed operator task must produce:

- [ ] Kernel code: `arrangement` + `application` + `make(...)`, repo style.
- [ ] A thin torch wrapper that pre-allocates the output and calls the kernel.
- [ ] `debug_arrangement` output confirming no OOB (`oob_count=0`) — captured in
      the trace log even if the result is trivially correct.
- [ ] Correctness test vs a PyTorch reference, ≥3 shapes × ≥2 dtypes, including
      one non-power-of-two shape; layout-sensitive ops also test a
      non-contiguous input.
- [ ] For perf-sensitive ops: a benchmark with baseline, ≥3 input sizes,
      throughput (GB/s) or TFLOPS, and a compute/memory-bound conclusion.
- [ ] For reduction / scatter-class ops: `inspect_generated_source.py
      --contract <contract>` output captured showing the required primitive is
      present in the generated source (a numerically green kernel can still
      be semantically wrong — e.g. race-prone plain-store scatter that
      passes on collision-free test data). Pick `<contract>` by op, not by
      family: reduction → `reduction` (softmax → `stable_softmax`);
      scatter / many-to-one → `atomic`; matmul → `matmul`; elementwise →
      `elementwise`. (`--contract layout`/`perf-diag` do not exist — those
      families map to one of the contracts above.)
- [ ] If a failure was hit: symptom → root cause → minimal fix → verification
      command and result.
- [ ] Explicit "not supported" note (dtype / dynamic shape / layout / hardware).
- [ ] [optional, human review] Arrangement PNG saved via
      `debug_arrangement.visualize_and_save(arrangement, tensors, save_dir=".")`.
      Requires `--with-viz` optional deps. Do NOT block task completion on this.

## 5. Cost Control / Constraints

- Do not re-read a reference file already in context; load each at most once.
- Do not inline reference content into the kernel or the report.
- Baseline for correctness is PyTorch (or an existing repo reference). Do not
  invent a third-party baseline beyond PyTorch / the repo's own Triton ops.
- Do not modify NineToothed compiler internals (`generation.py`, `jit.py`,
  `aot.py`, `auto_tuner.py`).
- Scripts are read-or-subprocess only: no `eval`, no `exec`, no dynamic
  `import` of untrusted code, no shell string interpolation, no network.
- Never delete or weaken tests to make them pass; never hard-code task names or
  expected answers.

## Quick API anchor (ninetoothed 0.25.0)

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor, block_size

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)   # user-passed constexpr
# block_size()                                       # auto-tuned tile size
# Symbol("X", meta=True)                             # auto-tuned meta value

def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

def application(input, output):
    output = input * 2  # noqa: F841   (assign to output param; F841 is expected)

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))
# invoke: out = torch.empty_like(x); kernel(x, out, BLOCK_SIZE=1024); return out
```

Generated Triton source is cached at `~/.ninetoothed/<sha256>.py`
(`ninetoothed.generation.CACHE_DIR`). Verify an arrangement without running a
real kernel via `ninetoothed.debugging.simulate_arrangement(arrangement,
tensors)`. See `references/perf-diag.md`.
