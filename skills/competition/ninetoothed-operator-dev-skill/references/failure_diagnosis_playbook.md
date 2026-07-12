# Failure Diagnosis Playbook

## Preserve The First Useful Failure

Record the exact command, working repository revision, return code, concise
stdout/stderr, and environment facts. Do not replace the original symptom with
a later secondary error.

## Classify

- Environment or dependency.
- Import or native-library loading.
- Test collection or fixture.
- Patch application or source revision mismatch.
- Generated source or compilation.
- Runtime launch or device selection.
- Shape, dtype, numerical, or layout correctness.
- Performance regression or noisy timing.

## Minimize

Reduce to one operator, one test node, one dtype, and one decisive shape. Keep
the failing semantics unchanged. Compare with a nearby passing operator when
that isolates shared infrastructure.

## Test A Root Cause

Form one falsifiable explanation. Inspect the file or environment fact that
would distinguish it, then run the smallest probe. Avoid stacking speculative
changes.

## Repair And Rerun

When the root cause and repair are within the task:

1. Make the smallest production or test change.
2. Rerun the exact reproducer.
3. Run the complete target operator test.
4. Inspect the diff and patch applicability.

## Patch Application Failure

Check, in order:

- LF versus CRLF line endings.
- Repository root and strip level.
- Recorded target revision.
- Stale context around each hunk.
- Already-applied or overlapping changes.
- Absolute or malformed patch paths.

Do not call an apply failure a runtime correctness failure. Regenerate from the
target repository root when source and intended change are known.

## Blocker Record

State the failed gate, evidence, dependency, exact next probe, and claims that
must remain unverified. Use `[BLOCKED]` only when the dependency truly prevents
meaningful progress.
