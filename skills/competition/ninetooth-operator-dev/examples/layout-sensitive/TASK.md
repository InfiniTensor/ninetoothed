# Self-Test Task: Layout-Sensitive Operator

## Input Task

Implement a NineToothed self-test operator named `strided_affine_copy` in a
temporary upstream test file such as `tests/test_strided_affine_copy.py`.

Semantics:

- input tensor `x`: rank 2, non-contiguous view with explicit row stride,
  column stride, and non-zero storage offset.
- output tensor `y`: rank 2, contiguous tensor with the same logical shape and
  dtype as `x`.
- operation: `y = x * 2.0 + 1.0`.
- layout requirement: the implementation must read the logical input values
  through stride-aware indexing or explicitly document a contiguous-only blocker.
- boundary behavior: cover at least one shape that does not divide evenly by the
  tile shape.
- dtype policy: `torch.float32` is required with `rtol=1e-5, atol=1e-6`;
  other dtypes are unsupported until tested.
- unsupported scope for this self-test: in-place writes into the strided input,
  overlapping storage, negative strides, rank other than 2, autograd behavior,
  and performance claims.

Required correctness cases:

- ordinary strided view: base shape `(80, 96)`, view shape `(37, 29)`, sliced as
  `base[2:76:2, 3:90:3]`;
- transposed view: `base.t()[5:42, 7:36]` or an equivalent non-contiguous
  transpose-backed view;
- small boundary view: logical shape `(1, 7)` with non-zero storage offset;
- output must compare against `x * 2.0 + 1.0`, not against the base tensor.

## Agent Execution Summary

Status: task specification only. No upstream patch, correctness run, benchmark
run, or performance conclusion has been recorded for this example yet.

Execution steps for the agent using this task:

1. Read the repository anchors below before designing the arrangement.
2. Add the smallest upstream self-test patch, preferably only
   `tests/test_strided_affine_copy.py`.
3. Start from the `test_clone.py` arrangement/application pattern because it
   already exercises non-contiguous strides through `data_ptr`, `offsets`, and
   `source.stride`.
4. Use explicit stride-aware load/store or `input.source[...]` indexing for the
   non-contiguous input. Do not insert `x.contiguous()` unless the task is
   intentionally recorded as blocked or partially complete.
5. Run the correctness command and record the real result in this file only
   after it has been executed.
6. Benchmark is optional for this category; record it only when the prompt asks
   for performance or when correctness passes and local device time permits.

## Repository Files Inspected

- `${NINETOOTHED_REPO}/tests/test_clone.py`: primary layout-sensitive
  reference for strided views, `data_ptr`, `offsets`, `source.stride`, and
  `input.source[...]` indexing.
- `${NINETOOTHED_REPO}/tests/test_data_ptr.py`: direct pointer access
  pattern and `ntl.atomic_add` example.
- `${NINETOOTHED_REPO}/tests/test_conv2d.py`: tiled layout transform with
  explicit `strides=...` in `tile`.
- `${NINETOOTHED_REPO}/tests/test_eval.py`: tensor offset evaluation,
  tiling, permuting, flattening, and tail sentinel behavior.
- `${NINETOOTHED_REPO}/src/ninetoothed/tensor.py`: `tile`, `offsets`,
  and stride metadata surface.
- `${NINETOOTHED_REPO}/src/ninetoothed/generation.py`: generated source
  handling for `data_ptr`, `offsets`, and `stride`.
- `${NINETOOTHED_REPO}/tests/utils.py`: device selection helper via
  `get_available_devices`.

## Patch Summary

Expected patch surface:

- add `tests/test_strided_affine_copy.py` in the upstream NineToothed checkout;
- do not edit NineToothed compiler internals;
- do not change existing clone, conv2d, or data pointer tests;
- do not commit generated cache files, benchmark artifacts, or raw logs.

Expected implementation shape:

- `strided_affine_copy(x)` allocates `torch.empty(x.shape, device=x.device,
  dtype=x.dtype)`;
- arrangement tiles `x` and `output` with a 2-D block shape;
- application loads logical values from the strided input using either
  `x.source.data_ptr() + x.offsets(0)[:, None] * x.source.stride(0) +
  x.offsets(1)[None, :] * x.source.stride(1)` or the closest
  `x.source[x.offsets(...)]` pattern from `test_clone.py`;
- application writes `value * 2.0 + 1.0` into the contiguous output tile;
- tests construct non-contiguous views and compare output to the PyTorch oracle.

## Correctness Command

```bash
cd ${NINETOOTHED_REPO}
python -m pytest tests/test_strided_affine_copy.py -q
```

## Correctness Result

Not run yet. Replace this line only with real pytest output or a concise summary
after the command above has been executed.

Required result fields after execution:

- command:
- environment:
- cases passed:
- failures:
- fix and rerun, if any:

## Benchmark Command

Benchmark is optional for this self-test category. If the prompt or reviewer
asks for performance evidence, use a like-for-like comparison after correctness
passes:

```bash
cd ${NINETOOTHED_REPO}
python - <<'PY'
import torch
import triton.testing

from tests.test_strided_affine_copy import strided_affine_copy
from tests.utils import get_available_devices

devices = get_available_devices()
if not devices:
    raise SystemExit("blocked: no available device")

device = devices[0]
dtype = torch.float32
base = torch.randn((4096, 8192), device=device, dtype=dtype)
x = base[3:4095:2, 5:8190:3]

candidate = lambda: strided_affine_copy(x)
baseline = lambda: x * 2.0 + 1.0

print(
    {
        "shape": tuple(x.shape),
        "stride": tuple(x.stride()),
        "storage_offset": x.storage_offset(),
        "dtype": str(dtype),
        "device": str(device),
        "candidate_ms": triton.testing.do_bench(candidate),
        "baseline_ms": triton.testing.do_bench(baseline),
    }
)
PY
```

## Benchmark Result

Not required by default. If the optional benchmark command is run, record the
real output, hardware/runtime blocker, or generated-source fallback evidence
here. Do not turn a blocked benchmark into a performance claim.

## Performance Conclusion

Out of scope unless the optional benchmark is executed. The required evidence
for this task is correctness over sliced, transposed, strided, or offset input.

## Failure Diagnosis

Not run yet. If correctness, generated-source inspection, or optional benchmark
execution fails, record:

- command:
- observed output:
- diagnosis path:
- root cause or blocker:
- minimal fix or workaround:
- rerun command:
- rerun result:

## Risks and Unsupported Scope

- This task intentionally checks non-contiguous correctness, not optimal memory
  coalescing.
- Overlapping storage, negative strides, rank other than 2, `torch.float16`,
  integer dtype, NaN/Inf policy, and autograd behavior are unsupported until
  explicit tests are added.
- A solution that calls `x.contiguous()` before launching the kernel does not
  satisfy the stride-aware requirement unless recorded as a blocker or fallback.
- Output layout is expected to be contiguous; writing back into a strided output
  belongs in a separate self-test.
- Benchmark evidence is optional here; if collected, it must include shape,
  stride, storage offset, dtype, device, and baseline.
