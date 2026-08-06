# Operator Task Contract

Use this contract before writing NineToothed operator code. It turns a natural
language task into explicit semantics, shape rules, verification evidence, and
change boundaries. If a field is unknown, write `unknown` and inspect the task
prompt or repository again before implementing.

The contract is intentionally a checklist, not a DSL tutorial. For repository
navigation use `repo-map.md`; for implementation patterns use
`dsl-pattern-index.md`; for test design use `verification-matrix.md`.

## Ready-to-Code Rule

Do not start implementation until these fields are no longer vague:

- semantic definition;
- input tensors and output tensors;
- shape rules;
- dtype rules;
- layout, stride, and offset assumptions;
- correctness reference;
- required tests;
- unsupported scope.

If the task prompt omits one of these, either infer it from an adjacent upstream
test or record it as an explicit assumption in the decision log.

## Requirement Fields

| Field | What to Capture | Required Evidence |
| --- | --- | --- |
| operator name | Public name, internal helper name, or task label. | Task prompt or matching upstream naming pattern. |
| semantic definition | Mathematical or tensor-level behavior, including element order and side effects. | PyTorch equivalent, existing NineToothed test, or concise formula. |
| input tensors | Each tensor name, rank, symbolic shape, layout expectation, and whether it may alias another tensor. | Prompt plus any matching test fixture. |
| output tensors | Output rank, shape relation to inputs, dtype, and whether output is newly allocated or written in place. | Prompt, reference implementation, or repository convention. |
| scalar or constexpr parameters | Scalars, compile-time constants, axis values, activation modes, epsilon values, tile sizes, or flags. | Prompt values and default policy when omitted. |
| shape rules | Rank constraints, dimension equations, batch/channel rules, reduce axes, and valid dynamic ranges. | At least one ordinary case and one boundary case. |
| dtype rules | Supported input/output dtypes, promotion rules, accumulator dtype, and tolerance implications. | Existing dtype coverage in similar tests or a stated task constraint. |
| broadcast rules | Which dimensions may broadcast, scalar behavior, singleton expansion, and disallowed broadcasts. | PyTorch reference or explicit shape examples. |
| mask or boundary behavior | Tail-block masking, out-of-bounds policy, empty or size-one dimensions, and non-power-of-two sizes. | Test cases that exercise a tail or boundary path. |
| contiguous assumptions | Whether inputs and outputs must be contiguous or may be views. | Upstream pattern or explicit unsupported-scope entry. |
| stride / offset / non-contiguous behavior | How load/store indices account for stride, storage offset, slicing, transposition, or fallback. | Required test or reason the task is contiguous-only. |
| numerical tolerance | `rtol`, `atol`, equality policy, NaN/Inf handling, and acceptable nondeterminism. | Reference test tolerance or dtype-specific rationale. |
| unsupported scope | Unsupported dtype, shape, layout, device, dynamic behavior, or performance promise. | Required for every task, even when the list is empty. |

## Acceptance Fields

| Field | What to Specify | Completion Standard |
| --- | --- | --- |
| correctness reference | PyTorch operation, existing NineToothed implementation, generated source expectation, or hand-computable oracle. | A reviewer can tell what output is correct without reading the final patch first. |
| required tests | Exact test file or test category, including ordinary, boundary, dtype, broadcast, and layout cases relevant to the task. | Tests cover every requirement field that affects correctness. |
| required benchmark | Whether benchmark evidence is mandatory, optional, or out of scope. Include input sizes and baseline when mandatory. | Performance-sensitive tasks name a baseline and at least two input scales. |
| generated source or AOT evidence | Whether to inspect generated code, AOT build output, or both. | Required for generation, integration, AOT, or performance-diagnosis tasks. |
| expected changed files | Smallest likely patch surface: operator, test, example, benchmark, docs, or diagnostic record. | Changes stay inside the named surface unless the decision log explains why. |
| forbidden changes | Files, APIs, tests, generated artifacts, broad formatting, or compiler-core areas that must not change. | No hidden expansion of scope during implementation. |
| verification commands | Commands to run for correctness, benchmark, generated source, AOT build, or structure checks. | Each command has an expected pass/fail interpretation before it is run. |
| result recording | Where to record command, environment, result, failures, fix, and rerun outcome. | No fabricated benchmark or test result; blockers are recorded as blockers. |

## Decision Log Fields

Record these before or during implementation so the final answer is auditable.

- similar repository files inspected: paths and why each was relevant.
- chosen implementation pattern: elementwise, broadcast, reduction, blocked,
  layout-sensitive, generated-source/AOT, fallback, or mixed.
- why this pattern fits: link the choice back to shape, dtype, layout, and
  verification requirements.
- assumptions made: every field inferred from context rather than explicitly
  stated in the task prompt.
- risks: correctness, performance, layout, dtype, numerical, integration, or
  environment risks that remain after the patch.
- fallback: smallest acceptable alternative if the preferred implementation,
  benchmark, or AOT path is blocked.
- files changed: final patch surface, kept separate from files only inspected.
- commands run: command, result, and whether a rerun was needed after a fix.

## Unsupported Scope Policy

`unsupported scope` is mandatory. Use `none identified` only after checking the
task prompt and the closest upstream pattern. Common entries include:

- dtype not covered by the task or upstream tests;
- dynamic rank or dynamic shape combinations not represented in examples;
- non-contiguous, strided, sliced, or offset tensors when the task is
  contiguous-only;
- device, architecture, or compiler mode not available in the local checkout;
- benchmark claims that cannot be reproduced in the current environment;
- numerical behavior beyond the stated tolerance.

Unsupported scope is not a workaround for missing correctness. If a required
case is unsupported, record the task as blocked or partially complete instead
of silently excluding it.

## Minimal Task Brief Template

```md
## Operator Requirement

- operator name:
- semantic definition:
- input tensors:
- output tensors:
- scalar or constexpr parameters:
- shape rules:
- dtype rules:
- broadcast rules:
- mask or boundary behavior:
- contiguous assumptions:
- stride / offset / non-contiguous behavior:
- numerical tolerance:
- unsupported scope:

## Acceptance

- correctness reference:
- required tests:
- required benchmark:
- generated source or AOT evidence:
- expected changed files:
- forbidden changes:
- verification commands:
- result recording:

## Decision Log

- similar repository files inspected:
- chosen implementation pattern:
- why this pattern fits:
- assumptions made:
- risks:
- fallback:
- files changed:
- commands run:
```
