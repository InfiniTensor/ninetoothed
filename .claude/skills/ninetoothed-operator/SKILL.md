---
name: ninetoothed-operator
description: Use when writing or converting operators to the Ninetoothed framework. Triggers include requests to implement Triton kernels using ninetoothed, create arrange/application functions, write operators like add/matmul/softmax/conv2d/attention for ninetoothed, or convert PyTorch-style operator descriptions into ninetoothed code.
---

# Ninetoothed Operator Writing Guide

## Overview

Ninetoothed lets you write GPU kernels using Python-like syntax instead of raw Triton. Every operator has two parts: `arrangement` (how to tile/split tensors) and `application` (the actual computation per tile). The framework handles pointer arithmetic, masking, and kernel launching automatically.

This skill uses an **iterative optimization loop**: the user must provide a CPU implementation as the precision baseline. The main Agent analyzes the CPU implementation and generates an initial Ninetoothed operator, then after Auto-tune, sequentially invokes the precision verification sub-Agent and the performance optimization sub-Agent, forming a **precision verification → performance optimization → iterative improvement** closed-loop workflow.

## When to Use

- User asks to implement an operator for the Ninetoothed framework
- User provides a CPU implementation and wants to convert it to a ninetoothed GPU kernel
- User provides a math expression, PyTorch equivalent, or computation description and wants a ninetoothed kernel
- User asks about `arrange`/`application` patterns, `Tensor`, `block_size()`, or `Symbol`

**Note: The user must provide a CPU implementation.** This code will serve as the sole baseline for precision verification. If the user does not provide a CPU implementation, the main Agent must proactively ask for one, explaining that it is a prerequisite for precision verification.

## Operator Classification Decision Tree

```
User request
  |
  +-- Element-wise (no reduction)?
  |     shape_in == shape_out, each output[i] = f(input[i])
  |     --> ELEMENT-WISE template
  |
  +-- Reduction along one or more dims?
  |     output shape < input shape, sum/max/min over dims
  |     --> REDUCTION template
  |
  +-- Matrix multiplication (dot product / GEMM)?
  |     C[M,N] = A[M,K] @ B[K,N], involves ntl.dot + loop
  |     --> MATMUL template
  |
  +-- Convolution?
  |     Sliding window + matmul decomposition
  |     --> CONV2D template (matmul + im2col arrangement)
  |
  +-- Pooling (max/avg over sliding window)?
  |     --> POOLING template
  |
  +-- Normalization (softmax, layernorm, rmsnorm)?
  |     --> REDUCTION + ELEMENT-WISE combination
  |
  +-- Attention (scaled dot-product)?
  |     --> FLASH ATTENTION template (online softmax + matmul)
```

## Code Templates

### Element-Wise

```python
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size
from ninetoothed.language import libdevice

BLOCK_SIZE = block_size()

def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

def application(input, output):
    output = libdevice.some_function(input)  # noqa: F841

tensors = (Tensor(1), Tensor(1))
kernel = ninetoothed.make(arrangement, application, tensors)

def op(input):
    output = torch.empty_like(input)
    kernel(input, output)
    return output
```

**Key rules:**
- All tensors tiled with same `BLOCK_SIZE`
- Use `block_size()` (not `Symbol(..., constexpr=True)`) for auto-tuning
- Scalar inputs (eps, scale): return as-is from arrangement, declare as `Tensor(0)`
- Assignment to output must use `# noqa: F841`
- Use `libdevice.*` for math functions (erf, tanh, sqrt, pow, exp, log, etc.)

**How `tile()` dimensions map to tensor dimensions:**

`tile()` zips its shape argument with the tensor's dimensions from the **first** dimension. This affects how you tile multi-dimensional tensors:

| Tensor shape | `tile((BLOCK_SIZE,))` | `tile((1, BLOCK_SIZE))` |
|---|---|---|
| 1D `(N,)` | Tiles dim 0 → `ceil(N/BLOCK_SIZE)` programs | N/A |
| 2D `(M, N)` | Tiles **dim 0** — processes `BLOCK_SIZE` rows, dim 1 fixed at 0 | Keeps dim 0 intact (1 program per row), tiles **dim 1** |
| 3D `(B, M, N)` | Tiles **dim 0** — processes `BLOCK_SIZE` batches | Keeps dim 0, tiles **dim 1** |

For 2D element-wise ops, use `tile((1, BLOCK_SIZE))` so each program processes one full row. For 3D batched ops, use `tile((1, 1, BLOCK_SIZE))`.

**When to upcast fp16 to fp32:**

Not all fp16 operations need upcasting. Only upcast when the computation is numerically sensitive:
- **Needs upcast**: softmax (exp + sum can overflow), layer norm (division by small variance), accumulation
- **Usually fine in fp16**: GELU, SiLU, sigmoid, erf, element-wise multiply/add

Test without upcasting first; only add it if precision doesn't meet requirements.

**Supporting arbitrary-dimensional inputs:**

`Tensor(ndim)` fixes the kernel to that many dimensions at compile time — the code generator only emits `ndim` size/stride parameters. Passing a tensor with more dimensions than declared causes incorrect pointer arithmetic. To support arbitrary dimensions, flatten in the wrapper and reshape the output:

```python
def gelu(input):
    original_shape = input.shape
    input = input.flatten()
    output = torch.empty_like(input)
    gelu_kernel(input, output)
    return output.reshape(original_shape)
```

### Reduction

```python
from ninetoothed import Tensor
from ntops.kernels.reduction import arrangement

def application(input, output):
    # input is (BLOCK_SIZE, original_dim) — loop over block
    result = ntl.zeros(output.shape, dtype=ntl.float32)
    for i in range(input.shape[0]):
        result += input[i]
    output = result  # noqa: F841

tensors = (Tensor(ndim, other=0), Tensor(ndim))  # other for padding fill
```

**Key rules:**
- Reduction dim becomes the inner dimension in the tiled tensor
- Use `ntl.sum(x, axis)` or manual loop depending on complexity
- `other` parameter: `float("-inf")` for max, `0` for sum/avg
- For multi-block reductions (e.g. softmax over large dim), use online algorithm

### Matrix Multiplication

```python
from ninetoothed import Tensor, block_size

BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()
BLOCK_SIZE_K = block_size()

def arrangement(input, other, output,
                BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K):
    output_arranged = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))
    input_arranged = input.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    other_arranged = other.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
    other_arranged = other_arranged.tile((-1, 1))
    other_arranged = other_arranged.expand((output_arranged.shape[0], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze(1)
    return input_arranged, other_arranged, output_arranged

def application(input, other, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(input.shape[0]):
        accumulator += ntl.dot(input[k], other[k])
    output = accumulator  # noqa: F841

tensors = (Tensor(2), Tensor(2), Tensor(2))
```

**Key rules:**
- `input.tile((1, -1))` + `.expand()` creates broadcast for reduction over K
- `.dtype.squeeze(0/1)` removes the singleton broadcast dimension from dtype hierarchy
- Always accumulate in fp32: `ntl.zeros(..., dtype=ntl.float32)`
- Use `block_size()` (not `Symbol(..., constexpr=True)`) for auto-tuning

### Softmax (Online Reduction)

```python
def application(input, output):
    input_loaded = input
    row_minus_max = input_loaded - ntl.max(input_loaded)
    numerator = ntl.exp(row_minus_max)
    denominator = ntl.sum(numerator)
    output = numerator / denominator  # noqa: F841
```

**Key rules:**
- Always subtract max for numerical stability
- Tile along the reduction dim: `tile((1, BLOCK_SIZE))` keeps row grouping
- `other=float("-inf")` on input tensor for padding

## Kernel Creation Patterns

### `ninetoothed.make()` — separate arrangement + application

Use when arrangement and application are defined as separate functions (most common pattern). See code templates above for examples.

### `@ninetoothed.jit` — inline type annotations

Use when you want a more concise style. Put the arrangement directly in the function signature as type annotations:

```python
import ninetoothed
from ninetoothed import Symbol, Tensor

def add(lhs, rhs):
    BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)

    @ninetoothed.jit
    def add_kernel(
        lhs: Tensor(1).tile((BLOCK_SIZE,)),
        rhs: Tensor(1).tile((BLOCK_SIZE,)),
        output: Tensor(1).tile((BLOCK_SIZE,)),
    ):
        output = lhs + rhs  # noqa: F841

    output = torch.empty_like(lhs)
    add_kernel(lhs, rhs, output)
    return output
```

**When to use which:** `make()` is preferred when arrangement is complex (matmul, attention) or reused across operators. `@jit` is preferred for simple operators where the arrangement is straightforward.

## Quick Reference: ntl Primitives

| Operation | Syntax | Notes |
|-----------|--------|-------|
| Zero tensor | `ntl.zeros(shape, dtype=...)` | For accumulators |
| Fill tensor | `ntl.full(shape, value, dtype=...)` | For initial values |
| Dot product | `ntl.dot(a, b)` | Matrix multiply on blocks |
| Transpose | `ntl.trans(x)` | Block transpose |
| Exp | `ntl.exp(x)` | Use fp32 intermediate for fp16 |
| Sigmoid | `ntl.sigmoid(x)` | Cast input to fp32 first |
| Rsqrt | `ntl.rsqrt(x)` | For normalization |
| Cast | `ntl.cast(x, dtype)` | Type conversion |
| Sum | `ntl.sum(x)` / `ntl.sum(x, axis)` | Reduction |
| Max | `ntl.max(x)` / `ntl.max(x, axis)` | Reduction |
| Where | `ntl.where(cond, a, b)` | Conditional select |
| Offsets | `x.offsets(dim)` | For mask manipulation |
| Method cast | `x.to(dtype)` | Same as ntl.cast |

## Quick Reference: libdevice Math Functions

Access via `from ninetoothed.language import libdevice`. These wrap `triton.language.extra.libdevice`.

| Operation | Syntax | Notes |
|-----------|--------|-------|
| Error function | `libdevice.erf(x)` | For GELU: `0.5 * x * (1 + erf(x / sqrt(2)))` |
| Hyperbolic tangent | `libdevice.tanh(x)` | For tanh-approx GELU |
| Square root | `libdevice.sqrt(x)` | Also available as `ntl.rsqrt(x)` for 1/sqrt |
| Power | `libdevice.pow(base, exp)` | General exponentiation |
| Exponential | `libdevice.exp(x)` | Also available as `ntl.exp(x)` |
| Natural log | `libdevice.log(x)` | Natural logarithm |
| Absolute value | `libdevice.abs(x)` | |
| Sine / Cosine | `libdevice.sin(x)` / `libdevice.cos(x)` | |
| Floor / Ceil | `libdevice.floor(x)` / `libdevice.ceil(x)` | |
| Minimum / Maximum | `ntl.minimum(a, b)` / `ntl.maximum(a, b)` | Element-wise min/max (maps to `tl.minimum`/`tl.maximum` in Triton) |

## Quick Reference: Tensor Meta-Operations

| Operation | Syntax | Purpose |
|-----------|--------|---------|
| Tile | `.tile(shape, strides, dilation)` | Create 2-level hierarchy (outer=grid, inner=block) |
| Expand | `.expand(shape)` | Expand singleton dims; `-1` = keep original |
| Squeeze | `.squeeze(dim)` | Remove singleton dims |
| Unsqueeze | `.unsqueeze(dim)` | Insert singleton dim |
| Permute | `.permute(dims)` | Reorder dims |
| Flatten | `.flatten(start_dim, end_dim)` | Flatten dims at one level |
| Ravel | `.ravel()` | Flatten entire hierarchy into one level |
| Pad | `.pad(pad)` | Add padding per dim |

## Quick Reference: Tensor Declaration

```python
Tensor(ndim)                    # Basic tensor
Tensor(ndim, other=float("-inf"))  # With padding fill value
Tensor(ndim, dtype=ninetoothed.float16)  # With explicit dtype
Tensor(ndim, shape_options={"constexpr": True})  # Compile-time shapes
Tensor(ndim, shape_options={"constexpr": True, "upper_bound": 128})  # Bounded constexpr
Tensor(0, constexpr=True, value=3.14)  # Scalar constant
```

## Quick Reference: Symbols

```python
from ninetoothed import Symbol, block_size

Symbol("BLOCK_SIZE", constexpr=True)     # User-specified compile-time constant
Symbol("NAME", meta=True)                # Auto-tuned meta parameter
Symbol("NAME", constexpr=True, upper_bound=128)  # Bounded constexpr
block_size()                             # Auto-tuned block size (preferred for tiling)
```

## Critical Constraints

1. **Outermost shapes must match**: All returned arranged tensors must have the same outermost shape (this defines the launch grid)
2. **Hierarchy depth >= 2**: After arrangement, tensors need at least 2 levels (outer=grid, inner=block)
3. **Assignment to output**: Use `output = expr` with `# noqa: F841` — never in-place ops
4. **Parameter names**: `arrangement` params must match `application` params exactly
5. **Tensor(0) for scalars**: Non-tensor inputs (eps, scale) use `Tensor(0)`

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forgetting `# noqa: F841` | Add it to every output assignment line |
| Missing `other=float("-inf")` for softmax | Set `other` on input Tensor for max-padding |
| Accumulating in wrong dtype | Always use `ntl.float32` for accumulators |
| Not subtracting max before exp | Causes overflow in softmax |
| Broadcasting mismatch in matmul | Use `.tile((1, -1))` + `.expand()` + `.dtype.squeeze()` pattern |
| Using `Symbol(..., constexpr=True)` for block sizes | Use `block_size()` instead for auto-tuning |
| Passing multi-dim tensor to `Tensor(1)` kernel | Use `flatten()`/`reshape()` in wrapper function |
| Using `ntl.*` for math when `libdevice.*` is needed | `ntl` has limited ops; use `libdevice.erf`, `libdevice.tanh`, etc. for full math library |
| Using `tile((BLOCK_SIZE,))` on 2D/3D tensor | `tile` zips from dim 0; use `tile((1, BLOCK_SIZE))` for 2D row-wise ops |
| Unnecessarily upcasting fp16 to fp32 | Only upcast for numerically sensitive ops (softmax, layer norm); test fp16 first |
| Outputting directly after code generation, skipping verification and optimization | Must execute the full 6-phase workflow: analyze → generate → auto-tune → precision verify → perf optimize → report |
| Introducing PyTorch or other third-party implementations as performance baseline | Performance optimization must only compare auto-tune baseline vs. optimized; third-party baselines are prohibited |
| Bypassing Ninetoothed framework to write raw Triton kernels | All optimizations must stay within the Ninetoothed framework; raw `@triton.jit` / `tl.*` usage is prohibited |
| Skipping optimization strategy priority order | Strategies 1-6 must be evaluated sequentially; no skipping allowed without explicit justification |
| Using "framework overhead" to justify bypassing the framework | Ninetoothed framework overhead is negligible (<1%) and is not a valid optimization target |

## Verification Checklist

- [ ] All arranged tensors have matching outermost shape
- [ ] Output assignment uses `# noqa: F841`
- [ ] Reduction uses `other` parameter on input Tensor
- [ ] Accumulators use `ntl.float32`
- [ ] `arrangement` parameter names match `application` parameter names
- [ ] Scalar inputs declared as `Tensor(0)`
- [ ] Softmax/max-pool subtracts max before exp
- [ ] fp16 inputs cast to fp32 before math operations
- [ ] `block_size()` used for auto-tunable tiling dimensions
- [ ] Multi-dimensional inputs use flatten/reshape pattern
- [ ] Math functions use `libdevice.*` when `ntl.*` doesn't suffice

## User Input Format

### Required: CPU Implementation Code

The user must provide a Python code snippet that implements the CPU version of the operator. This CPU implementation will serve as the **sole baseline** for precision verification.

Example:

```python
# User-provided CPU implementation example: matrix multiplication
def matmul_cpu(a, b):
    """Matrix multiplication CPU implementation"""
    M, K = a.shape
    K, N = b.shape
    c = torch.zeros(M, N, dtype=a.dtype)
    for i in range(M):
        for j in range(N):
            for k in range(K):
                c[i, j] += a[i, k] * b[k, j]
    return c
```

If the user does not provide a CPU implementation, the main Agent must proactively ask for one, explaining that it is a prerequisite for precision verification.

## Main Agent Responsibilities

### Analyze CPU Implementation

The main Agent must analyze the user-provided CPU code and extract the following information:

| Extraction Item | Description | Example |
|-----------------|-------------|---------|
| Operator type | elementwise / reduction / matmul / conv2d / softmax / attention | matmul |
| Number of inputs | Number of input tensors | 2 (a, b) |
| Number of outputs | Number of output tensors | 1 (c) |
| Dimension relationships | Relationships between dimensions | M,K * K,N -> M,N |
| Reduction operation | Whether accumulation/reduction exists | sum over k |
| Broadcast pattern | Whether broadcasting exists | None |

### Mandatory Execution Phases

**The main Agent must execute all of the following phases in order. No phase may be skipped, and the final result must not be output directly after code generation:**

| Phase | Description | Completion Criteria |
|-------|-------------|-------------------|
| 1. Analyze CPU implementation | Extract operator type, dimension relationships, reduction operations, etc. | Analysis complete, operator template determined |
| 2. Generate initial operator | Generate Ninetoothed code with `block_size()` | Code written to file |
| 3. Auto-tune | Automatically executed by ninetoothed's internal auto-tuner | Triggered on first kernel call, record `best_ms` |
| 4. Precision verification | Invoke precision verification sub-Agent, compare against user CPU implementation | Returns `status: "pass"` |
| 5. Performance optimization | Invoke performance optimization sub-Agent, self-iterative comparison | 3 consecutive improvements < 5% or max iterations reached |
| 6. Output report | Output complete report per Final Output template | Report includes precision + performance + code |

**Code generation (phase 2) is only the starting point of the entire workflow, not the end.** The main Agent must convert all phases into TodoWrite todo items before starting work, executing them step by step and marking each as complete.

### Generate Initial Ninetoothed Operator

Based on the CPU implementation analysis, the main Agent generates an initial implementation that conforms to Ninetoothed conventions. The code includes `block_size()` and other auto-tune markers, without presetting specific block sizes.

### Coordinate Iteration Loop

The main Agent executes the precision verification -> Auto-tune -> performance optimization iterative loop. See [Iterative Optimization Workflow](#iterative-optimization-workflow) for details.

## Shape Sampling

Performance tests must be run on specific shapes. When the user does not provide specific values, the skill should proactively sample representative shapes.

### Dimension Extraction

Automatically extract dimension names from the operator description and identify their roles:

| Operator Type | Description Example | Reduction Dims | Non-Reduction Dims |
|---------------|-------------------|----------------|-------------------|
| Matmul | `M,K * K,N -> M,N` | K | M, N |
| Element-wise | `M,N + M,N -> M,N` | None | M, N |
| Softmax | `M,N -> M,N` (reduce N) | N | M |
| Conv2d | `N,C,H,W * K,C,R,S -> N,K,H',W'` | C, R, S | N, K, H, W |

### Default Sampling Strategy

Automatically generate representative shape sets by operator type:

**Matmul (2D)** -- M=N=K in {64, 256, 512, 1024, 2048, 4096}

**Element-wise** -- Total element count in {64^2, 256^2, 512^2, 1024^2}

**Reduction (softmax etc.)** -- Reduction dimension in {64, 128, 256, 512, 1024}

| Category | Shape Examples | Purpose |
|----------|---------------|---------|
| Small | 64, 128, 256 | Verify small-scale computation correctness |
| Medium | 512, 1024 | Daily performance benchmark |
| Large | 2048, 4096 | Verify large-scale computation stability |
| Non-square | M=1024, N=512, K=2048 | Test unbalanced shapes |
| Extreme | M=1, N=4096, K=4096 | Edge case testing |

### User Override

Three interaction modes; the user can override the default sampling at any time:

```python
# Mode 1: Auto-sampling (default)
benchmark_matmul()

# Mode 2: Specify shape range -- Cartesian product combination
benchmark_matmul(shapes={"M": [64, 128, 256], "N": 256, "K": [128, 512]})

# Mode 3: Specify a single shape
benchmark_matmul(shapes={"M": 1024, "N": 2048, "K": 512})
```

### Output Format (when shapes not specified)

```
Warning: No input shapes specified. Representative shapes will be used for performance verification:

Matrix multiplication shape sampling (M, N, K):
- Small: (64, 64, 64)
- Medium: (256, 256, 256)
- Large: (1024, 1024, 1024)
- Large: (2048, 2048, 2048)
- Non-square: (1024, 512, 2048)

To customize shapes, pass them via the shapes parameter.
```

## Iterative Optimization Workflow

The CPU implementation is the sole precision baseline. Auto-tune handles parameter tuning. Performance optimization explores high-level optimization strategies on top of the Auto-tune optimal configuration. Precision takes priority -- performance optimization does not proceed when precision fails.

```
+-------------------------------------------------------------+
|                      Main Agent                               |
|  - Receive user-provided CPU implementation                   |
|  - Analyze CPU implementation, extract operator logic         |
|  - Generate initial Ninetoothed operator                      |
|  - Coordinate sub-Agent execution                             |
|  - Decide iteration flow                                      |
+-------------------------------------------------------------+
                              |
              +---------------+---------------+
              v                               v
+-----------------------------+   +-----------------------------+
|   Precision Verification    |   |  Performance Optimization   |
|   Sub-Agent                 |   |  Sub-Agent                  |
|   - Use user CPU impl as    |   |   - Receive auto-tuned      |
|     baseline                 |   |     operator                |
|   - Execute precision       |   |   - Optimize on top of      |
|     comparison               |   |     best config             |
|   - Return precision test   |   |   - Explore high-level      |
|     results                  |   |     optimization strategies  |
|                             |   |   - Return optimized code   |
|                             |   |     or mark no improvement  |
+-----------------------------+   +-----------------------------+
```

### Iteration Loop

```
+--------------------------------------------------+
|         Iteration loop start (iteration = 1)     |
+--------------------------------------------------+
                              |
                              v
+--------------------------------------------------+
| 1. Main Agent generates Ninetoothed operator     |
|    code                                           |
|    - Code includes block_size() etc. auto-tune   |
|      markers                                      |
|    - No preset block size                         |
+--------------------------------------------------+
                              |
                              v
+--------------------------------------------------+
| 2. Auto-tune executes automatically               |
|    - Iterate over candidate block size /          |
|      num_warps / num_stages                       |
|    - Select optimal configuration                 |
|    - Record best performance best_ms              |
+--------------------------------------------------+
                              |
                              v
+--------------------------------------------------+
| 3. Invoke precision verification sub-Agent        |
|    - Test using auto-tune selected optimal        |
|      configuration                                |
|    - Compare against user-provided CPU            |
|      implementation                                |
|    - Return precision verification results        |
+--------------------------------------------------+
                              |
                              v
                     +-----------------+
                     | Precision pass? |
                     +-----------------+
                              |
              +---------------+---------------+
              | No                           | Yes
              v                               v
      +-----------------+          +-----------------------------+
      | Fix precision   |          | 4. Invoke performance       |
      | issues, return  |          | optimization sub-Agent      |
      | to step 1       |          |   - Input: auto-tuned code  |
      +-----------------+          |   - Output: optimization     |
                                  |     result                   |
                                  +-----------------------------+
                                                  |
                                                  v
                                   +-----------------------------+
                                   | 5. Analyze optimization     |
                                   |    result                    |
                                   |    Improvement >= 5%?       |
                                   +-----------------------------+
                                                  |
                                  +---------------+---------------+
                                  | Yes                          | No
                                  v                               v
                          +-----------------+          +---------------------+
                          | Update code,    |          | 3 consecutive      |
                          | continue        |          | improvements < 5%? |
                          | iteration       |          +---------------------+
                          +-----------------+                     |
                                          +---------------+-------+
                                          | Yes          | No
                                          v               v
                                  +-----------------+ +-----------------+
                                  | Output final    | | Accept current  |
                                  | code and report | | code, continue  |
                                  +-----------------+ | iteration       |
                                                     +-----------------+
```

### Termination Conditions

**Successful termination:**
- Precision verification sub-Agent returns `status: "pass"` and performance optimization sub-Agent has 3 consecutive improvements < 5%
- Output final code and performance report

**Failure termination:**
- Precision verification fails 3 consecutive times -> exit and report precision issues
- Maximum iterations reached (default 10) -> output current code and performance report
- Performance optimization sub-Agent returns `status: "failed"` -> exit
- User manually interrupts

### Iteration Optimization Strategies

The main Agent should adopt the following optimization strategies based on sub-Agent feedback:

**Precision issue optimization (when precision does not pass):**
- Check if data type conversions are correct (e.g., fp16 accumulation requires fp32)
- Check precision accumulation in reduction operations
- Adjust algorithm implementation order (e.g., multiply before add)
- When precision does not pass, prioritize fixing precision issues; do not perform performance optimization

**Performance issue optimization (after precision passes, executed by performance optimization sub-Agent):**
- Optimize memory access patterns (coalesced access, reduce bank conflicts)
- Operator fusion (merge adjacent operations into a single kernel)
- Loop unrolling (manually unroll small loops)
- Reduce synchronization overhead (remove unnecessary synchronization points)
- Precision strategy adjustment (use lower precision when precision allows)
- Computation reorganization (change computation order to increase parallelism)

## Precision Verification

The precision verification sub-Agent uses the user-provided CPU implementation as the precision baseline, comparing the Ninetoothed operator output against the CPU implementation output.

### Input Format

```json
{
  "cpu_code": "def matmul_cpu(a, b): ...",
  "ninetoothed_code": "@nt.jit\ndef matmul_kernel(a, b, c): ...",
  "test_shapes": [
    {"M": 64, "N": 64, "K": 64},
    {"M": 256, "N": 256, "K": 256},
    {"M": 1024, "N": 1024, "K": 1024}
  ],
  "dtypes": ["float32", "float16"]
}
```

### Precision Testing Strategy

The precision verification sub-Agent must execute tests according to the following strategy.

### Tolerance Standards by Data Type

| Data Type | rtol | atol | Notes |
|-----------|------|------|-------|
| float32 | 1e-5 | 1e-5 | Standard precision |
| float16 | 1e-3 | 1e-3 | Lower precision |
| bfloat16 | 1e-2 | 1e-2 | Very low precision |
| int32/int64 | 0 | 0 | Must match exactly |
| bool | 0 | 0 | Must match exactly |

### Tolerance Standards by Operator Type

| Operator Category | rtol | atol | Special Notes |
|-------------------|------|------|---------------|
| element-wise (add, mul, relu) | 1e-5 | 1e-5 | Direct mapping, minimal error |
| matmul (fp32) | 1e-4 | 1e-4 | Accumulation error buildup |
| matmul (fp16) | 1e-2 | 1e-2 | Relaxed for lower precision |
| reduction (sum, max, min) | 1e-4 | 1e-4 | Floating-point accumulation error |
| softmax | 1e-4 | 1e-4 | Exponentiation amplifies error |
| layer_norm | 1e-4 | 1e-4 | Division and square root |
| attention | 1e-3 | 1e-3 | Complex computation chain |

### Required Checks

Precision tests must include the following checks:
1. `torch.allclose` passes with the tolerances above
2. No NaN values (`not torch.isnan().any()`)
3. No Inf values (`not torch.isinf().any()`)
4. Integer operations match exactly (`torch.equal()`)

### Precision Sub-Agent Output Format

```json
{
  "status": "pass" | "fail",
  "operator_name": "matmul",
  "test_results": [
    {
      "shape": {"M": 64, "N": 64, "K": 64},
      "dtype": "float32",
      "cpu_output_shape": [64, 64],
      "ninetoothed_output_shape": [64, 64],
      "max_absolute_error": 2.3e-6,
      "rtol_used": 1e-5,
      "atol_used": 1e-5,
      "passed": true,
      "has_nan": false,
      "has_inf": false
    }
  ],
  "overall_passed": true,
  "max_error_across_tests": 2.3e-6
}
```

## Performance Optimization

The performance optimization sub-Agent explores high-level optimization strategies on top of the Auto-tune optimal configuration.

### Core Responsibilities

- **Input**: Auto-tuned Ninetoothed operator code (with optimal block size, num_warps, num_stages)
- **Responsibility**: Explore high-level optimization strategies on top of the auto-tune optimal configuration
- **Output**: Optimized code, or mark as no significant improvement

### Prohibited Behaviors

The following actions are **strictly forbidden** during performance optimization:

- **Bypassing the Ninetoothed framework**: All optimizations must be implemented within the Ninetoothed framework (using `arrangement`/`application`, `ntl.*`, `libdevice.*`, etc.). Writing raw Triton kernels (`tl.*` directly, `@triton.jit`) is prohibited unless explicitly authorized by the user.
- **Skipping optimization priority order**: Strategies 1-6 must be evaluated in strict sequential order. No strategy may be skipped without explicit justification.
- **Using "framework overhead" as a reason to bypass**: The Ninetoothed framework overhead is negligible (<1%) and is not a valid optimization target.
- **Using third-party implementations as comparison baseline**: Performance comparison must only use the Ninetoothed operator's own before/after data.

### Performance Comparison Baseline

**Performance comparison must only use the Ninetoothed operator's own before/after optimization data. Introducing any third-party implementation as a comparison baseline is strictly prohibited.**

- **Baseline**: `best_ms` under the auto-tune optimal configuration (i.e., `original_best_ms`)
- **Comparison target**: `optimized_ms` after applying high-level optimization
- **Evaluation formula**: `improvement = (original_best_ms - optimized_ms) / original_best_ms * 100`

**Why third-party comparison is prohibited:**
1. This skill's performance optimization is **self-iterative optimization** -- selecting the best among different optimization strategies for the same operator
2. For unknown operators, a third-party implementation may not exist for comparison
3. The optimization level of third-party implementations is unknown; comparison results cannot guide optimization direction for this operator

### Explicitly Excluded Optimizations

The following optimizations are automatically handled by auto-tune. The performance optimization sub-Agent is **not responsible** for them:

| Parameter | Description | Handled By |
|-----------|-------------|------------|
| BLOCK_SIZE / BLOCK_M / BLOCK_N / BLOCK_K | Tile sizes | Auto-tune |
| num_warps | Number of warps per block | Auto-tune |
| num_stages | Pipeline stages | Auto-tune |

### Mandatory Optimization Strategy Checkpoint

**Before attempting any optimization, the performance optimization sub-Agent must output the following checkpoint and evaluate each strategy in strict sequential order (1 → 6). Each strategy must be explicitly marked as "evaluated" or "not applicable" with a brief reason before moving to the next.**

```
Optimization Strategy Checkpoint (must execute in order):
[ ] Strategy 1: Memory access pattern optimization — status: pending
[ ] Strategy 2: Operator fusion — status: pending
[ ] Strategy 3: Loop unrolling — status: pending
[ ] Strategy 4: Reduce synchronization overhead — status: pending
[ ] Strategy 5: Precision strategy adjustment — status: pending
[ ] Strategy 6: Computation reorganization — status: pending
```

**Exit condition**: Only after all 6 strategies have been evaluated and each has an explicit status with reason, may the sub-Agent conclude with "no_improvement". The final output must include the completed checkpoint and a declaration: "All high-level optimization strategies have been exhausted, no ≥5% improvement space found."

### Optimization Strategies (Strict Priority Order)

The performance optimization sub-Agent **must** evaluate strategies in the following order. Each strategy must be attempted or explicitly justified as inapplicable before proceeding to the next:

| Priority | Strategy | Description | How to Evaluate | Example |
|----------|----------|-------------|-----------------|---------|
| **1** | **Memory access pattern optimization** | Ensure coalesced access, reduce bank conflicts | Check if current tile/stride arrangement produces coalesced global memory access patterns; verify no shared memory bank conflicts | Adjust tensor layout, use vectorized loads |
| **2** | **Operator fusion** | Merge adjacent operations into a single kernel | Assess whether the operator can be fused with its most common upstream/downstream operations to reduce kernel launch overhead and memory traffic | Fuse add + relu into a single pass |
| **3** | **Loop unrolling** | Manually unroll small loops | For operators with small fixed-count loops (e.g., 2-3 iterations), evaluate if manual unrolling reduces loop overhead | Unroll 2-iteration reduction loops |
| **4** | **Reduce synchronization overhead** | Remove unnecessary synchronization points | Review the kernel for unnecessary `tl.sync()` calls or barrier operations that could be eliminated | Remove redundant sync in element-wise ops |
| **5** | **Precision strategy adjustment** | Use lower precision when precision allows | Check if intermediate computations currently done in fp32 could safely use fp16/bf16 without exceeding precision tolerances (requires re-running precision verification) | Change fp32 accumulator to fp16 |
| **6** | **Computation reorganization** | Change computation order to increase parallelism | Reorder operations to enable better instruction-level parallelism or reduce data dependencies | Do partial reduction before multiplication |

### Input Format

```json
{
  "ninetoothed_code": "@nt.jit\ndef matmul_kernel(a, b, c): ...",
  "auto_tune_info": {
    "best_config": {
      "BLOCK_M": 128,
      "BLOCK_N": 128,
      "BLOCK_K": 32,
      "num_warps": 8,
      "num_stages": 3
    },
    "best_ms": 0.512,
    "configs_explored": 12
  },
  "test_shapes": [
    {"M": 1024, "N": 1024, "K": 1024}
  ]
}
```

### Performance Improvement Evaluation

```python
def evaluate_optimization(original_best_ms, optimized_ms):
    """
    Determine if optimization succeeded.
    - original_best_ms: Performance under auto-tune optimal configuration
    - optimized_ms: Performance after applying high-level optimization
    """
    if optimized_ms is None:
        return "failed"

    improvement = (original_best_ms - optimized_ms) / original_best_ms * 100

    if improvement >= 5.0:
        return "optimized"   # Optimization succeeded, use optimized code
    else:
        return "no_improvement"  # No significant improvement, keep original code
```

### Output Format

```json
{
  "status": "optimized" | "no_improvement" | "failed",
  "original_best_ms": 0.512,
  "optimized_ms": 0.478,
  "improvement_percent": 6.6,
  "improved_code": "@nt.jit\ndef matmul_kernel(a, b, c): ...",
  "optimization_applied": "vectorized memory loads",
  "auto_tune_config_unchanged": true,
  "strategy_checkpoint": {
    "1_memory_access": {"status": "evaluated", "result": "already optimal, coalesced access confirmed"},
    "2_operator_fusion": {"status": "evaluated", "result": "standalone operator, no fusion opportunity"},
    "3_loop_unrolling": {"status": "not_applicable", "reason": "no loops in element-wise operator"},
    "4_sync_overhead": {"status": "not_applicable", "reason": "no synchronization points"},
    "5_precision_adjustment": {"status": "not_applicable", "reason": "already using optimal precision"},
    "6_computation_reorg": {"status": "evaluated", "result": "optimized via vectorized loads"}
  }
}
```

## Final Output

### Success Output

```markdown
## Operator Generation Successful

### Final Ninetoothed Code
[Final optimized operator code]

### Precision Verification Results
Baseline: User-provided CPU implementation

- Status: PASS
- Test shapes: 5
- Test data types: float32, float16
- Max absolute error: 2.3e-6

### Auto-tune Results

| Config | Best Value | Candidates |
|--------|------------|------------|
| BLOCK_M | 128 | 12 |
| BLOCK_N | 128 | 12 |
| BLOCK_K | 32 | 12 |
| num_warps | 8 | 4 |
| num_stages | 3 | 3 |
| Best performance | 0.512 ms | - |

### High-Level Optimization History

| Iteration | Strategy | Before (ms) | After (ms) | Improvement |
|-----------|----------|-------------|------------|-------------|
| 1 | Vectorized loads | 0.512 | 0.498 | 2.7% |
| 2 | Operator fusion | 0.498 | 0.478 | 4.0% |
| 3 | No significant improvement | - | - | - |

### Final Performance
- Best performance: 0.478 ms
- Improvement over auto-tune optimal: 6.6%
- Final code includes all optimizations
```

### Failure Output

```markdown
## Operator Generation Failed

### Final Code
[Last attempted code]

### Failure Reason
- Precision verification failed 3 consecutive times / max iterations reached (10)
- Last precision result: FAIL (max_error=1.5e-3, exceeds tolerance 1e-4)
- Last performance result: optimized (6.6% improvement) / no_improvement

### Test Results Summary
[Detailed test data]

### Recommendations
- Consider using higher-precision intermediate accumulation types
- Check edge case handling in the CPU implementation
```

## Configuration

```yaml
iterative_config:
  max_iterations: 10                       # Maximum iteration count
  consecutive_precision_failures: 3        # Consecutive precision failure threshold
  consecutive_no_improvement: 3            # Consecutive no-improvement threshold
  performance_improvement_threshold: 5.0   # Performance improvement threshold (%)

precision_config:
  default_rtol: 1e-5
  default_atol: 1e-5
  by_dtype: {...}                  # Tolerances by data type
  by_op_type: {...}                # Tolerances by operator type

optimization_config:
  excluded_from_optimization:      # These parameters are handled by auto-tune
    - BLOCK_SIZE
    - BLOCK_M
    - BLOCK_N
    - BLOCK_K
    - num_warps
    - num_stages
```

## Detailed Reference

For complete API documentation, repository analysis, and advanced patterns, read @reference.md.

For detailed step-by-step examples, read @examples.md.
