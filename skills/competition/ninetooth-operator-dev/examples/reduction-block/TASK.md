# Self-Test Task: Reduction / Blocked Operator

## Input Task

Implement a NineToothed self-test operator named `row_softmax_tail` in a
temporary upstream test file such as `tests/test_row_softmax_tail.py`.

Semantics:

- input tensor `x`: rank 2, shape `(num_rows, num_cols)`, contiguous.
- output tensor `y`: rank 2, same shape and dtype as `x`.
- reduce axis: last dimension only, equivalent to `torch.softmax(x, dim=-1)`.
- block/tile shape: one row by one column block group, using a symbolic or
  constexpr `BLOCK_SIZE_N` that covers the row segment being reduced.
- numerical stability: subtract the row max before exponentiation, then divide
  by the row sum.
- tail behavior: lanes outside `num_cols` must be masked with `-inf` before the
  max/sum path so non-power-of-two columns do not affect the result.
- dtype policy: `torch.float32` is required with `rtol=1e-4, atol=1e-5`;
  `torch.float16` may be added only with an explicit tolerance rationale.
- unsupported scope for this self-test: non-contiguous input, reductions over
  dimensions other than the last axis, dynamic rank, integer dtype, in-place
  output, and backward/autograd behavior.

Required correctness cases:

- ordinary dense shape: `(32, 128)`;
- non-power-of-two tail shape: `(17, 781)`;
- small boundary shape: `(3, 7)`;
- numerically shifted input, for example `x * 8 - 4`, to exercise stability.

## Agent Execution Summary

Status: task specification only. No upstream patch, correctness run, benchmark
run, or performance conclusion has been recorded for this example yet.

Execution steps for the agent using this task:

1. Read the repository anchors below before designing the arrangement.
2. Add the smallest upstream self-test patch, preferably only
   `tests/test_row_softmax_tail.py`.
3. Mirror the row-wise reduction shape from upstream softmax first; use
   max-shift-exp-sum normalization rather than a direct `exp(x) / sum(exp(x))`
   path.
4. Ensure the arrangement or tensor metadata uses a masked `other=float("-inf")`
   path for tail lanes.
5. Run the correctness command and record the real result in this file only
   after it has been executed.
6. Run the benchmark command or record a blocker with device/runtime details.

## Repository Files Inspected

- `${NINETOOTHED_REPO}/tests/test_softmax.py`: row-wise softmax
  reference, `other=float("-inf")`, stable max/exp/sum pattern, and
  non-power-of-two column coverage.
- `${NINETOOTHED_REPO}/tests/test_max_pool2d.py`: reduction over a tiled
  window, `ntl.max(input, axis=1)`, `floor_mode`, and output dtype squeezing.
- `${NINETOOTHED_REPO}/tests/test_matmul.py`: explicit block symbols and
  output tile shape conventions.
- `${NINETOOTHED_REPO}/tests/test_attention.py`: blocked reduction loop,
  masked out-of-range lanes, and stable online normalization pattern.
- `${NINETOOTHED_REPO}/tests/utils.py`: device selection helper via
  `get_available_devices`.

## Patch Summary

Expected patch surface:

- add `tests/test_row_softmax_tail.py` in the upstream NineToothed checkout;
- do not edit NineToothed compiler internals;
- do not modify unrelated reduction tests;
- do not commit generated cache files, benchmark artifacts, or raw logs.

Expected implementation shape:

- `row_softmax_tail(x)` allocates `torch.empty_like(x)`;
- `@ninetoothed.jit` or `ninetoothed.make` mirrors the nearest upstream
  reduction pattern;
- input tile uses `Tensor(2, other=float("-inf"))` or an equivalent masking
  route for out-of-range tail lanes;
- application computes the row maximum with `ntl.max(input_row)` or the nearest
  equivalent axis form from upstream tests, then `row_minus_max`,
  `numerator = ntl.exp(row_minus_max)`, and
  `output = numerator / ntl.sum(numerator, axis=...)`;
- tests compare against `torch.softmax(x, dim=-1)` for all required shapes.

## Correctness Command

```bash
cd ${NINETOOTHED_REPO}
python -m pytest tests/test_row_softmax_tail.py -q
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
import triton.testing

from tests.test_row_softmax_tail import row_softmax_tail
from tests.utils import get_available_devices

devices = get_available_devices()
if not devices:
    raise SystemExit("blocked: no available device")

device = devices[0]
dtype = torch.float32
sizes = ((32, 128), (1823, 781), (2048, 4097))

for rows, cols in sizes:
    x = torch.randn((rows, cols), device=device, dtype=dtype) * 8 - 4

    candidate = lambda: row_softmax_tail(x)
    baseline = lambda: torch.softmax(x, dim=-1)

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

- This task intentionally validates contiguous rank-2 row-wise softmax only;
  layout-sensitive and multi-axis reductions belong in separate self-tests.
- `torch.float16`, very large columns beyond local memory limits, NaN/Inf
  policy, and backward/autograd behavior are unsupported until specific tests
  are added.
- Tail masking is correctness-critical: out-of-range lanes must not influence
  the max or denominator.
- Benchmark conclusions must mention device, dtype, cache state, warmup/repeat
  defaults from `triton.testing.do_bench`, and all input sizes.
- If benchmark execution is blocked, inspect the generated source or kernel
  arrangement for obvious redundant reduction work, but record that as fallback
  evidence rather than a timing claim.
