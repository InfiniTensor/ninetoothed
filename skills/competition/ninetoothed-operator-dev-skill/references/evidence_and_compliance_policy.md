# Evidence And Compliance Policy

## Status Vocabulary

- `[VERIFIED]`: direct command, log, artifact, or inspected file supports it.
- `[INFERRED]`: code reading or reasoning supports it, but it was not exercised.
- `[TODO-GPU]`: a compatible device runtime is required.
- `[BLOCKED]`: a named prerequisite prevents the next meaningful check.

Use status on claims, not as decoration. State the command or artifact close to
the claim it supports.

## Required Boundaries

- Skipped tests are not passes.
- Collection is not runtime correctness.
- File discovery is not generated-source success.
- Compilation is not dispatch or correctness.
- One selected-shape ratio is not broad speedup.
- A local apply-check does not rewrite a historical server outcome.
- One layout example is not full non-contiguous support.

## Safety

- Do not package credentials, server addresses, private keys, tokens, or private
  user paths.
- Do not include hidden evaluation answers, task-name bypasses, or evaluator
  detection logic.
- Do not delete, weaken, skip, or bypass tests.
- Do not fabricate output, screenshots, timings, or hardware facts.
- Disclose external repositories, dependencies, licenses, and AI assistance.

## Evidence Packaging

Prefer compact raw CSV, patch, JSON, or command output plus a short README.
Redact private connection details while preserving technical facts. Record
source revision and hashes so evidence can be tied to the intended source.

## Handoff

Report changed files, commands, results, unfavorable comparisons, blockers,
unsupported behavior, and the smallest remaining user action. Keep Git push,
pull request creation, paid infrastructure, and signature under human control.
