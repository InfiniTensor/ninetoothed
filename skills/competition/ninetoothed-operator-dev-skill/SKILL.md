---
name: ninetoothed-operator-dev-skill
description: >-
  Implement, test, debug, benchmark, optimize, and integrate NineToothed or
  ntops operators. Use for arrangement and application work, wrappers and
  exports, correctness tests, broadcasting, reductions, stride or offset
  layouts, generated source, AOT builds, performance regressions, failing
  tests, minimal patches, and InfiniCore dispatch. Do not use for unrelated
  general programming, ordinary prose, or non-NineToothed repositories.
---

# NineToothed Operator Development

Build the smallest correct repository-native change, prove what the current
environment can prove, and label every remaining boundary honestly.

- Version: `1.0.0`
- Contest: `2026-spring-T3-1-1`
- Compatibility: offline-capable; Python and Git required; CUDA, PyTorch,
  Triton, NineToothed, and ntops are optional until runtime validation is due.

## Trigger Scope

Use this skill for:

- Implementing or changing a NineToothed or ntops operator.
- Adding an arrangement, application, tensor meta-operation, load, or store.
- Updating a public wrapper, export, dtype path, mask, or boundary rule.
- Writing correctness tests against PyTorch or a repository reference.
- Fixing broadcast, reduction, blocking, stride, offset, or layout behavior.
- Diagnosing a failing test, generated source, AOT build, or integration path.
- Designing a correctness-gated benchmark or investigating a regression.
- Verifying an InfiniCore `use_ntops` dispatch path and its fallback.
- Producing a minimal patch and repository-ready integration summary.

Do not use this skill for:

- General Python functions unrelated to NineToothed.
- Unrelated repositories, ordinary chat, weather, or business copy.
- Compiler-core redesign unless the user explicitly asks for it.
- Claims about hardware or runtime paths that have not actually run.

## Nonnegotiable Rules

- Inspect the repository before editing.
- Implement when the request asks for a change; stop at analysis only when the
  user explicitly requests analysis.
- Preserve repository style, helper APIs, and ownership boundaries.
- Keep the patch narrow; avoid broad refactors and formatting churn.
- Never delete, weaken, skip, or bypass tests to obtain a pass.
- Never fabricate logs, timings, generated artifacts, or device conclusions.
- Keep source, test, and benchmark evidence tied to the same revision.
- Treat skipped tests as skipped, not passed.
- Mark unavailable device work as `[TODO-GPU]` and blocked work as `[BLOCKED]`.
- Do not access online services or private credentials unless the user grants
  that exact authority.

## 1. Discover The Repository

1. Run `git rev-parse --show-toplevel` from the likely repository.
2. Record `git status --short`, branch, and `git rev-parse HEAD`.
3. Read `README`, contribution guidance, package metadata, and test config.
4. Locate nearby operators with `rg --files` and symbol searches.
5. Read the closest kernel, wrapper, export, test, and example together.
6. Check local instructions before choosing commands or dependencies.

Use [repository reading routes](references/repo_reading_routes.md) when the
operator family or integration boundary is unclear.

## 2. Write A Requirement Card

Capture before coding:

- Mathematical semantics and PyTorch reference.
- Inputs, outputs, ranks, shapes, and dynamic dimensions.
- Dtypes, promotion, accumulation dtype, and tolerance.
- Broadcasting, mask, axis, keepdim, and reduction domains.
- Padding, dilation, stride, storage offset, and contiguity assumptions.
- Empty, singleton, tail, boundary, overflow, and numerical-stability cases.
- Supported and unsupported devices or layouts.
- Required wrapper, export, test, benchmark, and integration changes.
- Acceptance commands and evidence expected from each command.

Resolve ambiguity from repository precedent. Ask the user only when competing
interpretations materially change public behavior.

## 3. Select The Closest Pattern

- Compare at least one nearby operator with the same computation family.
- Follow existing arrangement and application composition.
- Reuse tensor meta-operations and load/store helpers already used nearby.
- Match wrapper signatures, exports, test parameterization, and skip policy.
- Prefer a local helper over a new abstraction unless duplication is material.
- Keep unsupported behavior explicit instead of silently broadening scope.

Read the family guide that applies:

- [Elementwise and broadcast](references/elementwise_broadcast_guide.md)
- [Reduction and blocking](references/reduction_blocking_guide.md)
- [Layout, stride, and offset](references/layout_stride_offset_guide.md)

## 4. Implement The Minimal Correct Change

1. Change the production kernel or wrapper needed by the requirement card.
2. Update exports only when adding or exposing a public operator.
3. Add the smallest correctness test that fails before and passes after.
4. Cover the decisive dtype, shape, mask, broadcast, or layout boundary.
5. Avoid unrelated cleanup, renaming, and repository-wide formatting.
6. Review the diff immediately and remove accidental scope growth.

When a failing test exposes an in-scope root cause, fix it and rerun the exact
same command. Do not stop after describing an obvious, safe repair.

## 5. Prove Correctness

1. Run import or collection checks first when environment setup is uncertain.
2. Run the narrowest changed test file or test node.
3. Compare with PyTorch or the repository's accepted reference implementation.
4. Exercise representative normal, boundary, and failure cases.
5. Add layout cases only to the support level actually required.
6. Run the relevant operator family after the focused test passes.
7. Record command, revision, device, result counts, and skipped cases.

Read [correctness testing](references/testing_correctness_guide.md) for
tolerances, parameterization, CUDA gates, and failure interpretation.

## 6. Handle Layout Deliberately

- Distinguish logical shape from physical storage.
- Inspect strides and storage offset before assuming contiguous indexing.
- Verify view, transpose, slice, and offset behavior separately when relevant.
- Derive output shape from padding, dilation, stride, and kernel size.
- Do not claim full non-contiguous support from one transposed or sliced case.
- If the implementation intentionally requires contiguous input, enforce and
  document that boundary consistently in code and tests.

## 7. Benchmark Only After Correctness

1. Establish a passing correctness guard on the timed inputs.
2. Pin baseline, candidate, revision, device, shape, dtype, and layout.
3. Separate compile or first-call time from steady-state runtime.
4. Synchronize the device around timing.
5. Record warmup, repeat, raw samples, mean, median, and minimum.
6. Compare on the same process and hardware where practical.
7. Report variance and selected-shape limits.
8. Do not generalize a one-shape ratio into broad speedup.

Use [performance benchmarking](references/performance_benchmark_guide.md) and
the bundled `scripts/run_ntops_microbenchmark.py` for controlled short runs.

## 8. Verify Generated Source, AOT, And Dispatch

- Record the source revision and generation command.
- Inspect the generated artifact itself, not only the build script.
- Compile or load the artifact before claiming build success.
- Run a correctness path that consumes the generated output.
- For InfiniCore, import the native package before checking `use_ntops`.
- Prove the flag, supported device guard, selected route, and fallback behavior.
- Keep generated-source, AOT, dispatch, correctness, and timing claims separate.
- Stop dependent timing when import, dispatch, or correctness is blocked.

Read [generated source, AOT, and integration](references/generated_source_aot_integration_guide.md).

## 9. Close Failures

1. Reproduce with the smallest stable command.
2. Classify environment, import, compile, runtime, numerical, layout,
   applicability, or performance failure.
3. Preserve the first useful error and relevant environment facts.
4. Test one root-cause hypothesis at a time.
5. Apply the smallest in-scope fix.
6. Rerun the exact reproducer and then the relevant family test.
7. Record unresolved dependencies as `[BLOCKED]` with a concrete next check.

Use the [failure diagnosis playbook](references/failure_diagnosis_playbook.md).

## 10. Produce An Applicable Patch

- Generate the patch from the target repository root.
- Keep repository-relative `a/` and `b/` paths and LF line endings.
- Run `git diff --check`, `git diff --stat`, and `git diff --name-only`.
- Validate with `git apply --check` against the recorded clean revision.
- Apply and test in an isolated copy when the patch is a handoff artifact.
- Report stale context, wrong strip level, revision mismatch, or line-ending
  failure precisely; never relabel a failed apply-check as correctness failure.

Use [patch applicability](references/patch_applicability_guide.md) and
`scripts/validate_patch_artifact.py`.

## 11. Report Evidence Precisely

- `[VERIFIED]`: supported by an included command output or artifact.
- `[INFERRED]`: reasoned from inspected code but not directly exercised.
- `[TODO-GPU]`: needs a compatible device runtime.
- `[BLOCKED]`: cannot proceed until a named prerequisite changes.
- Include baseline strengths when they outperform the skill-guided run.
- Include candidate weaknesses and unsupported cases without euphemism.
- Never claim hidden-task success, broad speedup, generated output, AOT, or
  dispatch without direct evidence.

Read [evidence and compliance](references/evidence_and_compliance_policy.md).

## 12. Finish The Task

Before handing off:

1. Inspect `git status --short` and the complete diff.
2. Run focused tests and the broadest relevant affordable test.
3. Run correctness before any timing command.
4. Validate patch applicability against the intended revision.
5. State files changed, commands run, results, blockers, and evidence paths.
6. Leave only actions that genuinely require the user, such as credentials,
   paid infrastructure approval, signature, push, or pull request creation.

## Reference Index

Start at [references/index.md](references/index.md), then open only the guide
needed for the current operator family or failure mode.
