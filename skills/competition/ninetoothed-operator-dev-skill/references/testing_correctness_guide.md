# Correctness Testing Guide

## Build The Oracle

Prefer PyTorch or the repository's accepted reference implementation. Match the
same input dtype, axis semantics, broadcasting, padding, and output dtype.
Avoid reimplementing the candidate algorithm as the oracle.

## Test Progression

1. Import or collect the target test.
2. Run one smallest representative case.
3. Run the newly added regression case.
4. Run the complete operator test file.
5. Run the nearby operator family when the shared helper changed.

Use the same command before and after the repair when diagnosing a failure.

## Parameter Selection

Select cases that distinguish implementations rather than maximizing count:

- Normal and boundary shapes.
- Aligned and non-aligned tails.
- Dtypes with distinct promotion or tolerance.
- Broadcast or axis variants that alter indexing.
- Supported layout classes and one enforced unsupported class.
- Numerically difficult values when the operation is sensitive.

## Tolerance

Use repository conventions first. Base `rtol` and `atol` on dtype and algorithm,
not on the need to make a failing output pass. Diagnose the maximum error and
its location before widening a tolerance.

## CUDA And Skips

- Record the device and dependency probe.
- Treat unavailable CUDA as `[TODO-GPU]`.
- Report passed, failed, skipped, expected-failed, and unexpected-passed counts.
- Do not translate collection success into runtime correctness.
- Do not translate an import success into operator correctness.

## Evidence Record

Keep the exact command, revision, start/end status, concise output, and any
environment prerequisite. Redact private connection details before packaging.

## Failure Review

Classify shape, dtype, numerical, layout, environment, or stale-test failure.
Make the smallest in-scope repair and rerun the exact failing command.
