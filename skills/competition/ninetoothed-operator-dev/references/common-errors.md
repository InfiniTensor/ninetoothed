# Common errors → root cause → minimal fix

> Provenance: entries are **[S]** (derived from the 0.25.0 source and the
> error-characterisation methodology cited in REFERENCE.md) unless marked
> **[E]** — observed live and fixed during this project's GPU episode runs
> (2026-07-10/11; raw run logs kept with the companion harness, not committed
> here).

## NEVER (anti-patterns)

- Do not loosen tolerance (raise atol/rtol) to force a pass — fix the root cause via the fp32 upcast in error #5.
- Do not delete or skip a failing test case.
- Do not refactor unrelated code while fixing one operator (hurts the patch-minimality score).
- Do not compare CPU PyTorch against GPU NineToothed — always compare on the same device (GPU vs GPU).
- Do not declare correctness from a single shape/dtype — cover at least 3 shapes x 2 dtypes.
- Do not let reductions (sum/softmax/norm) accumulate below fp32 — this directly causes error #5.
- Do not optimize performance before correctness passes.

## Error-characterisation table (quantify the error first, then find the cause)

Borrowed from Ascend agent-skills: look at the data distribution before reading the code.

| Symptom | Most likely root cause | See |
|---|---|---|
| fp16 fails, fp32 passes | accumulation not upcast to fp32 | #5 |
| all-zero output (max_abs ~ 0) | output tensor never written | #6 / #2 |
| NaN / Inf present | exp overflow (no stability) / divide by zero | #4 |
| uniform deviation, cosine_sim ~ 1 | systematic precision loss (wrong scale / truncation) | #5 |
| periodic / striped errors | tile boundary or stride/offset miscomputation | #3 / #10 |
| only tail elements wrong | tail-tile alignment or mask handling | #4 |
| results differ across runs | missing sync (add `torch.cuda.synchronize()`) | — |
| small shape passes, large shape fails | tiling boundary coverage error | #6 / #10 |

**MERE/MARE thresholds** (pass when MERE < threshold AND MARE < 10 x threshold):

| dtype | threshold |
|---|---|
| float32 | 1.22e-4 (2^-13) |
| float16 | 9.77e-4 (2^-10) |
| bfloat16 | 7.81e-3 (2^-7) |
| int / bool | exact match |

Compute directly with `run_correctness_matrix.py --mere-check got.pt ref.pt`.

---

## Error index

Symptom-first index. For each: the likely root cause and the smallest change.
Always re-run the correctness matrix after a fix.

| # | Symptom | Root cause | Minimal fix |
|---|---------|------------|-------------|
| 1 | `TypeError: ... not iterable` from arrangement | arrangement returned a bare `Tensor`, not a tuple | add trailing comma / wrap: `return x.tile((BS,)),` |
| 2 | Lint `F841 local variable 'output' assigned but never used` | the `output = ...` write is the kernel's effect; lint can't see it | append `# noqa: F841` to each output-assignment line |
| 3 | Wrong values on a 2-D op, off by rows | used `tile((BLOCK_SIZE,))` (tiles dim 0) for a per-row op | use `tile((1, BLOCK_SIZE))` for row-wise |
| 4 | NaN / Inf from softmax or max | missing padding fill on the input | declare `Tensor(2, other=float("-inf"))` for the reduced input |
| 5 | fp16/bf16 result off by tolerance on sum/softmax/norm | accumulation done in low precision | `ntl.cast(x, ntl.float32)` before the sum, cast result back |
| 6 | `RecursionError` building the kernel | output arrangement forced to mirror input's block hierarchy | arrange the output independently (e.g. `tile((1,))` per row) |
| 7 | eval error when using `unsqueeze` in arrangement | some versions can't eval `unsqueeze` there | reshape in the wrapper (`view`/`unsqueeze`) before the kernel |
| 8 | Multi-dim tensor fails a `Tensor(1)` kernel | rank mismatch | `flatten()` in the wrapper, `view`/`reshape` the result back |
| 9 | `block_size()` / `Symbol(constexpr=True)` confusion | wrong knob: constexpr is user-passed, `block_size()`/`meta=True` are auto-tuned | user-tunable → `Symbol("X", constexpr=True)`; let framework tune → `block_size()` or `Symbol("X", meta=True)` |
| 10 | matmul wrong / shape error on K reduction | K-dim broadcast not set up | use the double-tile: `.tile((1,-1)).expand((-1, N)).dtype.squeeze(0)` (and symmetric for the other operand) |
| 11 | AOT build fails but JIT worked | `num_warps`/`num_stages` defaults mismatch target | pass them explicitly to `make(..., caller="cuda", num_warps=, num_stages=)` |
| 12 | Can't find generated source | kernel not built yet, or wrong digest | build/run once, then `inspect_generated_source.py` (newest) |
| 13 | Non-contiguous input gives wrong result | kernel assumed contiguous storage | wrapper fast-path `x = x.contiguous()`, or express layout via `tile(strides=)` / `permute` (layout.md) |
| 14 | Task needs data-dependent indexing / atomic scatter / value-dependent control flow / dynamic output shape | not a bug — outside the DSL's expressible set (0.25.0) | STOP iterating; declared PyTorch fallback with the `# ninetoothed-fallback:` marker (SKILL.md §3 `dsl_limit`) |

## Diagnosis loop (use when symptom not in table or MERE/MARE table)

1. **Reproduce** with the smallest failing shape/dtype from the matrix.
2. **Localize**: is it arrangement (use `simulate_arrangement`) or application
   (check the generated source)?
3. **Hypothesize one cause**, apply the smallest change. If the cause turns
   out to be error #14 (inexpressible semantics), classification is
   `dsl_limit` — skip straight to the declared fallback; more attempts
   cannot converge.
4. **Re-verify** with `run_correctness_matrix.py`; if still failing after 3
   attempts on the same cause, stop and report a PyTorch fallback + the
   diagnosis (declared with the `# ninetoothed-fallback:` marker, per
   SKILL.md Stop Rules).

## What NOT to do

- Do not loosen test tolerance to force a pass (fix #5 instead).
- Do not delete or skip a failing test.
- Do not refactor unrelated code while fixing one operator (hurts the
  "patch minimality" rubric item).
