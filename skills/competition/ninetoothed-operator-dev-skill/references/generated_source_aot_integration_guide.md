# Generated Source, AOT, And Integration Guide

## Generated Source

1. Record the source revision and generation command.
2. Capture the generated filenames and hashes.
3. Inspect the generated kernel, signature, launch parameters, and guards.
4. Compile or load the artifact.
5. Run correctness through a path that consumes that artifact.

Finding a generator script or an old output file is not runtime verification.

## AOT Build

- Identify the smallest operator-specific generation path.
- Confirm required compiler, toolkit, and architecture configuration.
- Keep build output separate from source scans.
- Require a successful exit code and expected artifact.
- Link or load the result before claiming the build is usable.
- Avoid a broad build when only a narrow path is authorized.

If a minimal operator-only command is unavailable, explain the scope and cost
before running a repository-wide build.

## InfiniCore Dispatch

Prove each gate in order:

1. Import `infinicore` and its native library.
2. Import ntops and the requested public operator.
3. Confirm the `use_ntops` flag state.
4. Confirm the supported device guard.
5. Exercise the public InfiniCore call.
6. Compare output with PyTorch.
7. Show route evidence and fallback behavior.

Use `scripts/probe_infinicore_dispatch.py` for a compact probe. A missing
native module blocks dispatch, correctness, and dependent timing.

## Claim Separation

Track generated source, AOT compilation, loading, dispatch, correctness, and
performance as separate claims. One successful gate does not imply the next.

## Common Failures

- Source revision differs from the patch target.
- Generated artifact is stale or absent.
- Architecture flags do not match the device.
- Native Python extension is not installed.
- `use_ntops` is false or the device guard selects fallback.
- Timing proceeds after correctness or dispatch was blocked.
