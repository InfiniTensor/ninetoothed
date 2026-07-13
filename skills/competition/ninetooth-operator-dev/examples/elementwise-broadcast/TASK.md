# Self-Test Task: Elementwise / Broadcast

## Input Task

Implement a NineToothed self-test operator named `bias_relu` in a temporary
upstream test file such as `tests/test_bias_relu.py`.

Semantics:

- input tensor `x`: rank 2, shape `(num_rows, num_cols)`, contiguous.
- input tensor `bias`: rank 1, shape `(num_cols,)`, contiguous.
- output tensor `y`: rank 2, same shape and dtype as `x`.
- operation: `y = torch.relu(x + bias[None, :])`.
- broadcast: `bias` broadcasts across the row dimension only.
- boundary behavior: cover a non-power-of-two shape so tile tails and masks are
  exercised.
- dtype policy: `torch.float32` is required with `rtol=1e-5, atol=1e-6`;
  `torch.float16` may be added only if the local device and upstream tolerance
  pattern support it.
- unsupported scope for this self-test: non-contiguous views, in-place output,
  dynamic rank, scalar bias, and dtype claims beyond tested dtypes.

Required correctness cases:

- ordinary dense shape: `(64, 128)`;
- broadcast tail shape: `(37, 257)`;
- small boundary shape: `(1, 17)`;
- bias values containing negative and positive entries so both ReLU branches are
  exercised.

## Agent Execution Summary

Status: task specification only. No upstream patch, correctness run, benchmark
run, or performance conclusion has been recorded for this example yet.

Execution steps for the agent using this task:

1. Read the repository anchors below before designing the arrangement.
2. Add the smallest upstream self-test patch, preferably only
   `tests/test_bias_relu.py`.
3. Use a row/column tile arrangement for `x` and `output`; tile `bias` over the
   column dimension, then use `unsqueeze`/`expand` or the closest repository
   pattern to align it with the output tile.
4. Use `ninetoothed.language` operations already present in upstream tests,
   such as `ntl.maximum` or `ntl.where`, for ReLU behavior.
5. Run the correctness command and record the real result in this file only
   after it has been executed.
6. Run the benchmark command or record the blocker. Do not infer a performance
   result from correctness alone.

## Repository Files Inspected

- `${NINETOOTHED_REPO}/tests/test_add.py`: direct 1-D elementwise
  `@ninetoothed.jit` pattern and `torch.allclose` style.
- `${NINETOOTHED_REPO}/tests/test_pow.py`: `ninetoothed.make`
  elementwise arrangement/application split.
- `${NINETOOTHED_REPO}/tests/test_expand.py`: minimal `expand` usage.
- `${NINETOOTHED_REPO}/tests/test_generation.py`: scalar constexpr,
  shape-changing arrangement, and dtype metadata adjustment examples.
- `${NINETOOTHED_REPO}/tests/test_matmul.py`: broadcast-by-expand pattern
  that aligns one tiled dimension with an output tile.
- `${NINETOOTHED_REPO}/tests/test_attention.py`: use of
  `ninetoothed.language` elementwise functions such as `ntl.maximum` and
  `ntl.where`.
- `${NINETOOTHED_REPO}/tests/utils.py`: device selection helper via
  `get_available_devices`.

## Patch Summary

Expected patch surface:

- add `tests/test_bias_relu.py` in the upstream NineToothed checkout;
- do not edit NineToothed compiler internals;
- do not edit unrelated tests or apply broad formatting;
- do not commit generated cache files, benchmark artifacts, or local logs.

Expected implementation shape:

- `arrangement(x, bias, output, BLOCK_SIZE_M=..., BLOCK_SIZE_N=...)`;
- `x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))`;
- `bias.tile((BLOCK_SIZE_N,))` aligned to the output columns and expanded across
  rows;
- `output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))`;
- `application(x, bias, output)` computes `output = max(x + bias, 0)` using a
  repository-style NineToothed language expression.

## Correctness Command

```bash
cd ${NINETOOTHED_REPO}
python -m pytest tests/test_bias_relu.py -q
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

Run only after the correctness command passes.

```bash
cd ${NINETOOTHED_REPO}
python - <<'PY'
import torch
import triton
import triton.testing

from tests.test_bias_relu import bias_relu
from tests.utils import get_available_devices

devices = get_available_devices()
if not devices:
    raise SystemExit("blocked: no available device")

device = devices[0]
dtype = torch.float32
sizes = ((64, 128), (1024, 4097))

for rows, cols in sizes:
    x = torch.randn((rows, cols), device=device, dtype=dtype)
    bias = torch.randn((cols,), device=device, dtype=dtype)

    candidate = lambda: bias_relu(x, bias)
    baseline = lambda: torch.relu(x + bias[None, :])

    candidate_ms = triton.testing.do_bench(candidate)
    baseline_ms = triton.testing.do_bench(baseline)
    print(
        {
            "shape": (rows, cols),
            "dtype": str(dtype),
            "device": str(device),
            "candidate_ms": candidate_ms,
            "baseline_ms": baseline_ms,
        }
    )
PY
```

## Benchmark Result

Not run yet. Record the real benchmark output, hardware/runtime blocker, or
generated-source fallback evidence here. A blocked benchmark is acceptable only
if it names the attempted command, device/runtime blocker, intended baseline,
and input sizes.

## Performance Conclusion

Pending real benchmark evidence. Do not claim a speedup, slowdown, or parity
until the benchmark command has run successfully or a blocker has been recorded.

## Failure Diagnosis

Not run yet. If correctness, generated-source inspection, or benchmark execution
fails, record:

- command:
- observed output:
- diagnosis path:
- root cause or blocker:
- minimal fix or workaround:
- rerun command:
- rerun result:

## Risks and Unsupported Scope

- This task intentionally validates contiguous rank-2 input plus rank-1 trailing
  bias broadcast; layout-sensitive views belong in the layout-sensitive
  self-test.
- `torch.float16`, integer, boolean, NaN, and Inf behavior are unsupported until
  specific tests and tolerances are added.
- The benchmark compares against PyTorch eager `torch.relu(x + bias[None, :])`;
  conclusions must mention cache state, device, dtype, and input sizes.
- If bias expansion needs dtype metadata adjustments, mirror the closest
  upstream pattern instead of inventing a broad helper.
- Generated source or AOT evidence is optional for this elementwise task unless
  the benchmark is blocked or performance is unexpectedly poor.
