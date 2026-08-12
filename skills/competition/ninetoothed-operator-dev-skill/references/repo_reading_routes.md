# Repository Reading Routes

## Establish Context

Run from the target repository:

```bash
git rev-parse --show-toplevel
git status --short
git rev-parse HEAD
rg --files
```

Read local instructions, `README`, contribution guidance, package metadata,
test configuration, and the closest operator before editing.

## Operator Route

1. Find the public wrapper or call site by symbol name.
2. Follow imports to the kernel arrangement and application.
3. Identify tensor meta-operations, load/store functions, and boundary masks.
4. Find the export list and nearby operators with the same family.
5. Read the corresponding tests, shared fixtures, skip rules, and tolerances.
6. Inspect examples or benchmark utilities only when the task needs them.

Useful searches:

```bash
rg -n "operator_name|premake|make\(" src tests examples
rg -n "pytest.mark.parametrize|allclose|skip" tests
rg -n "generated|build_ntops|use_ntops" .
```

Replace `operator_name` with the requested symbol. Adapt directory names to the
repository instead of assuming a fixed checkout layout.

## Integration Route

For wrapper or dispatch work, trace both directions:

- Public Python API to wrapper, kernel, generated output, and device launch.
- Device registration back to dispatcher, feature flag, and public API.

Do not infer runtime dispatch from file presence. Require import, flag, route,
correctness, and fallback evidence.

## Exit Criteria

Before editing, be able to name:

- The production file that owns the behavior.
- The nearest accepted implementation pattern.
- The test file that proves the requested behavior.
- The command that should fail before and pass after.
- Any device, dtype, layout, or build prerequisite.
