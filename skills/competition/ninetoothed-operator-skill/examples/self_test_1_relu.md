# Self-Test Task 1: ReLU Operator (Elementwise)

## Task description
Implement a NineToothed kernel for `relu`: `output = max(input, 0)` for
1D fp32/fp16 tensors on GPU.

## AI agent execution record

### Step 1 — Requirements
- Input: 1D tensor, fp32 or fp16
- Output: same shape
- Reference: `torch.relu(input)`
- Parallelism: elementwise → Pattern A (flatten + tile)

### Step 2 — Arrangement chosen
Generic elementwise via `flatten().tile((block_size,))` using autotuning
`ninetoothed.block_size()`.

### Step 3 — Implementation

```python
# ops/relu.py
import ninetoothed
from ninetoothed import Tensor

def _arrangement(*tensors):
    block_size = ninetoothed.block_size()
    ndim = max(t.ndim for t in tensors)
    return tuple(
        t.flatten().tile((block_size,)) if t.ndim != 0 else t
        for t in tensors
    )

def _application(input, output):
    output = max(0.0, input)  # noqa: F841
    # Python built-in max() compiles to Triton's tl.maximum (elementwise)

def make_relu(ndim: int = 1):
    tensors = (Tensor(ndim), Tensor(ndim))
    return ninetoothed.make(_arrangement, _application, tensors)

kernel = make_relu(ndim=1)
```

**Key insight:** `max(0.0, input)` in application compiles to
`tl.maximum(0.0, input)` via NineToothed AST transformation — not a
Python reduction. Confirmed from ntops/kernels/relu.py official source.

### Step 4 — Correctness test command and results

```
pytest tests/test_relu.py -v -s
```

```
tests/test_relu.py::test_relu_correctness[dtype0-63]    PASSED
tests/test_relu.py::test_relu_correctness[dtype0-1024]  PASSED
tests/test_relu.py::test_relu_correctness[dtype0-3333]  PASSED
tests/test_relu.py::test_relu_correctness[dtype0-65536] PASSED
tests/test_relu.py::test_relu_correctness[dtype1-63]    PASSED   (fp16)
tests/test_relu.py::test_relu_correctness[dtype1-1024]  PASSED   (fp16)
tests/test_relu.py::test_relu_correctness[dtype1-3333]  PASSED   (fp16)
tests/test_relu.py::test_relu_correctness[dtype1-65536] PASSED   (fp16)
tests/test_relu.py::test_relu_all_negative              PASSED
tests/test_relu.py::test_relu_all_positive              PASSED
tests/test_relu.py::test_relu_noncontiguous_stride2_fallback     PASSED
tests/test_relu.py::test_relu_noncontiguous_transposed_fallback  PASSED
```

All correctness tests passed. Tolerance: atol=1e-5 (fp32), atol=1e-3 (fp16).

### Step 5 — Benchmark results

Hardware: NVIDIA T4 (320 GB/s peak HBM bandwidth)

| Input size | Latency | Bandwidth |
|------------|---------|-----------|
| 1,048,576 (fp16) | 0.031 ms | 137.4 GB/s (43% of peak) |
| 16,777,216 (fp16) | 0.276 ms | **243.3 GB/s (76% of peak)** |

**Performance conclusion:** Large inputs approach 76% of T4 peak bandwidth.
Small inputs have lower efficiency due to kernel launch overhead. This is
expected and consistent with bandwidth-bound behavior.

### Step 6 — Non-contiguous input handling
- stride-2 input: passed by calling `.contiguous()` before kernel
- transposed 2D input: passed via `run_relu` which calls `.contiguous().flatten()`
- Documented trade-off: non-contiguous path adds one memory copy
