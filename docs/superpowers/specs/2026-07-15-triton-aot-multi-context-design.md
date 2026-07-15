# Triton AOT Multi-Context Design

## Goal

Allow one Triton AOT handle, including a reloaded built artifact, to launch in
multiple CUDA contexts without reusing a `CUmodule` or `CUfunction` from a
different context.

## Scope

- Keep the current backend-selection behavior unchanged.
- Preserve the generated `<kernel>_kernel_default` ABI.
- Support repeated context transitions such as `0 -> 1 -> 0` and
  `1 -> 0 -> 1`.
- Keep previously built single-context artifacts loadable only when they carry
  the new context guard; older cached binaries are rebuilt during
  materialization.
- Do not address Triton's generated `dev = 0` shared-memory query in this
  change.

## Root Cause

Triton's AOT `compile.c` template stores one generated `CUmodule` and
`CUfunction` in process-global variables. NineToothed loads the resulting
shared library once and captures one exported function. The first launch
initializes those globals in the current CUDA context. A later launch in a
different context reuses the first context's handles and returns
`CUDA_ERROR_INVALID_HANDLE`.

The historical NineToothed AOT implementation avoided this by keying kernel
state by `CUcontext`. The SSA materializer rewrite removed that behavior while
retaining the multi-device regression test.

## Architecture

`_compile_aot_library` will strictly discover the low-level module/function
symbol pairs in Triton's generated C sources. It will then generate a small
NineToothed-owned C++ companion and link it into the same shared library.

The companion exports two stable functions:

- `ninetoothed_triton_enter`: lock the library-local launch mutex, query the
  current `CUcontext`, load or restore all generated module/function pairs for
  that context, and leave the mutex held.
- `ninetoothed_triton_leave`: release the mutex after the generated launcher
  has submitted its asynchronous kernel launch.

The state map is keyed by `CUcontext`, not a device ordinal. A state contains
all generated `CUmodule` and `CUfunction` values for the artifact. Loading a
new context sets the generated globals to null, invokes Triton's generated
load functions, then records the resulting handles. Returning to a known
context restores its recorded handles.

The mutex spans `enter -> generated launcher -> leave` because the generated
globals are shared by all calls into one DSO. It serializes only host-side
launch submission; it does not synchronize or serialize GPU execution.

Both fresh materialization and `load_built_artifact` resolve the guard exports
and pass them to the same Python AOT wrapper. The wrapper always calls
`leave` in a `finally` block after a successful `enter`.

## Compatibility And Cache Handling

Symbol discovery accepts only the expected Triton declarations and fails with
a clear error if the generated template changes. It must never silently link
an unguarded binary.

The Triton AOT manifest carries a launcher schema value. Under the existing
artifact cache lock, a missing or stale schema forces recompilation before the
binary is published. This prevents an existing unsafe `.triton.so` from
surviving the code upgrade under an unchanged compilation cache key.

Reloading a built artifact requires the companion exports. An artifact without
them reports that it must be rebuilt instead of silently retaining the
multi-context bug.

## Error Handling

- A failed or null current-context query returns a CUDA driver error from
  `enter`; Python raises `KernelLaunchError` without invoking the kernel.
- The generated kernel's nonzero result remains a `KernelLaunchError`.
- `leave` runs after kernel errors and Python exceptions once `enter`
  succeeds.
- Symbol-discovery failures identify the unsupported Triton AOT source format.

## Tests

CPU-only tests will verify:

- generated module/function/load symbols are discovered strictly;
- malformed or changed generated sources fail explicitly;
- the companion source contains a `CUcontext` state map and stable exports;
- the companion is included in the final NVCC input list;
- fresh and reloaded wrappers call `enter`, kernel, and `leave` in order;
- `leave` runs on kernel errors and exceptions;
- a stale launcher schema requests recompilation.

The real GPU regression will explicitly compile `backend="triton"`, then run
fresh and reloaded handles through `0 -> 1 -> 0` and `1 -> 0 -> 1`. Each launch
will synchronize and compare its output. The existing
`tests/test_aot.py::test_add[True-45327-dtype0-bf16-cuda]` remains the original
end-to-end regression.

## Non-Goals

- Changing `caller="cuda"` backend inference.
- Copying one DSO per device or context.
- Unloading context modules during device switches.
- Supporting heterogeneous GPU architectures in one compiled artifact.
