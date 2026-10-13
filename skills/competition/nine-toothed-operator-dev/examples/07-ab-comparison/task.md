# Self-Test 07: Skill A/B Comparison (softmax-temperature)

## Category

Meta-evaluation: measure the behavioral difference of the same agent on the
same NineToothed operator task, with and without this skill installed.

## Protocol

Identical task text given to both conditions (see `ab_protocol.md` in the
session workspace): implement row-wise `output[i, :] = softmax(input[i, :] / T)`,
T scalar float (default 1.0), 2-D fp16/fp32 CUDA input, correctness vs
`torch.softmax(input / T, dim=-1)`, plus a benchmark.

- **A (no skill)**: task text + repo access only.
- **B (with skill)**: must read `SKILL.md` first, then the same task.

Same model, same tool access, same repo snapshot (master c9ebd49), same GPU
(RTX 4090, torch 2.9.1+cu128, Triton 3.5.1).

## GPU Execution Results (both artifacts run on the pod)

| Condition | Correctness | Benchmark (NT/torch time ratio, lower = faster) |
|---|---|---|
| A | 8/8 allclose, `ALL CORRECTNESS PASSED` | 0.50x–0.75x across 4 shape/dtype points |
| B | 9/9 allclose (incl. non-contiguous transposed view), `ALL CORRECTNESS PASSED` | 0.51x–0.74x across the same 4 points |

Both beat the unfused `input / T` + softmax baseline (fusion removes one
elementwise kernel + temp allocation). Performance is statistically
indistinguishable between conditions — the skill's value shows up in
process and coverage, not raw speed on a task this close to an existing
repo pattern.

## Scored Behaviors (checklist from the protocol)

| # | Behavior | A | B |
|---|---|---|---|
| 1 | Restates contract (shape/dtype/T semantics/boundary) before coding | Partial — design notes only, no upfront contract | Yes — dedicated "Task contract" section per skill step 1 |
| 2 | Searches nearest existing pattern before writing | Yes (test_softmax.py) | Yes (routed via SKILL.md -> operator-patterns.md -> test_softmax.py) |
| 3 | `other=float("-inf")` tail fill | Yes | Yes |
| 4 | Scalar T as `Tensor(0)`, not tiled | Yes* | Yes |
| 5 | fp32 accumulation for fp16 input | Yes | Yes |
| 6 | Avoids module-level constants inside application | Yes* | Yes |
| 7 | Correctness test incl. non-power-of-two row length | Yes (1823x781) | Yes (1823x781) |
| 8 | Benchmark with command/sizes/baseline/conclusion | Yes | Yes |
| 9 | States unsupported cases | Yes (list) | Yes (asserted in wrapper, with per-case failure-mode reasons) |
| 10 | Minimal patch, no unrelated changes | Yes | Yes |

Score: A 8.5/10, B 10/10.

\* Contamination caveat, disclosed below.

## Behaviors Observed Only in Condition B

- Explicit contract restatement (dtype rules, layout rules, boundary
  behavior, unsupported cases) before any code.
- A ninth correctness check on a **non-contiguous transposed 2-D view**,
  motivated by the api-notes stride-awareness/rank-match rule. A never
  tested non-contiguous input.
- Chose `torch.empty(input.shape, ...)` over `torch.empty_like(input)`,
  citing the api-notes `empty_like`-on-views pitfall, and defined the
  output-contiguity contract accordingly.
- Considered and **rejected** the `Tensor(0, constexpr=True)` alternative
  for T with a reason (per-value recompilation), citing
  `test_generation.py::test_non_int_constexpr`.
- A "Residual risk" section with concrete one-line fallbacks (e.g.
  `ntl.cast` -> `.to(ntl.float32)` if the former fails on the GPU box).
- Per-case failure-mode reasons for unsupported inputs (why T <= 0 breaks
  the -inf tail fill, why rank != 2 silently misprocesses).

## Contamination Caveat (disclosed)

Condition A's own log records that it read `selftest_gelu.py` and
`selftest_l2norm.py` from the session workspace. Those artifacts embed
skill-derived lessons: the module-level-constants NameError pitfall
(documented inside selftest_gelu.py after self-test 05's failure loop) and
the `Tensor(0)` scalar-arg + 2-D reduction combination (selftest_l2norm.py).
A's success on behaviors 4 and 6 is therefore partially attributable to
indirect skill leakage. A clean-room A would plausibly have hit the
NameError compile failure that self-test 05 hit before the skill documented
it. This biases the comparison **toward A**, so the measured gap is a lower
bound.

## Conclusion

On a task with a close in-repo precedent, both conditions produce correct,
fast code. The skill's measured contribution: complete contract discipline
(behavior 1), non-contiguous coverage that A missed entirely, pitfall-cited
design choices (`empty_like`, constexpr-vs-runtime scalar), and residual-risk
documentation with fallbacks. On hidden tasks *without* a close precedent —
or on layout-sensitive tasks where the untested non-contiguous path is the
grading target — these are exactly the behaviors the rubric rewards.

Artifacts: `ab_protocol.md`, `ab_test_a_impl.py`, `ab_test_b_impl.py`,
`logs/ab_test_a.md`, `logs/ab_test_b.md`, `logs/ab_run_a.log`,
`logs/ab_run_b.log` (session workspace).
