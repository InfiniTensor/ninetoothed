# Ninetoothed Operator Examples

## Example 1: Element-Wise Add (Simple)

**Input**: Add two 1D tensors element-wise.

```python
import ninetoothed
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(input, other, output, BLOCK_SIZE=BLOCK_SIZE):
    input_arranged = input.tile((BLOCK_SIZE,))
    other_arranged = other.tile((BLOCK_SIZE,))
    output_arranged = output.tile((BLOCK_SIZE,))

    return input_arranged, other_arranged, output_arranged


def application(input, other, output):
    output = input + other  # noqa: F841


tensors = tuple(Tensor(1) for _ in range(3))

kernel = ninetoothed.make(arrangement, application, tensors)
```

**Walkthrough:**
1. Three 1D tensors: two inputs, one output
2. `arrangement` tiles each into blocks of `BLOCK_SIZE` — all outermost shapes are `(ceil(N/BLOCK_SIZE),)`, matching the grid constraint
3. `application` receives one block at a time; `input + other` is element-wise addition on Triton tensors
4. Assignment to `output` triggers a store

---

## Example 2: SiLU Activation (Element-Wise + Type Cast)

**Input**: SiLU(x) = x * sigmoid(x), with fp16 precision.

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))


def application(input, output):
    input_loaded = input
    output = input_loaded * ntl.sigmoid(ntl.cast(input_loaded, ntl.float32))  # noqa: F841


tensors = (Tensor(1), Tensor(1))
kernel = ninetoothed.make(arrangement, application, tensors)
```

**Walkthrough:**
1. Single input, single output — both 1D
2. `ntl.cast(input_loaded, ntl.float32)` upcasts before sigmoid for numerical stability
3. The result is automatically cast back to the output dtype

---

## Example 3: SwiGLU (Element-Wise with Two Inputs)

**Input**: SwiGLU(a, b) = a * SiLU(b) = a * b * sigmoid(b)

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(a, b, c, BLOCK_SIZE=BLOCK_SIZE):
    return a.tile((BLOCK_SIZE,)), b.tile((BLOCK_SIZE,)), c.tile((BLOCK_SIZE,))


def application(a, b, c):
    b_loaded = b
    gate = b_loaded * ntl.sigmoid(ntl.cast(b_loaded, ntl.float32))
    c = a * gate  # noqa: F841


tensors = (Tensor(1), Tensor(1), Tensor(1))
kernel = ninetoothed.make(arrangement, application, tensors)
```

**Walkthrough:**
1. Three tensors: two inputs (a, b), one output (c)
2. Compute gate = b * sigmoid(b), then c = a * gate
3. Naming convention: inputs can be named descriptively (a, b) instead of generic (input, other)

---

## Example 4: RMS Norm (Row Reduction + Element-Wise)

**Input**: RMSNorm(x, eps) = x * rsqrt(mean(x^2) + eps)

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(input, eps, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((1, BLOCK_SIZE)), eps, output.tile((1, BLOCK_SIZE))


def application(input, eps, output):
    input_fp32 = ntl.cast(input, ntl.float32)
    output = input_fp32 * ntl.rsqrt(  # noqa: F841
        ntl.sum(input_fp32 * input_fp32) / input.shape[-1] + eps
    )


tensors = (Tensor(2), Tensor(0), Tensor(2))
kernel = ninetoothed.make(arrangement, application, tensors)
```

**Walkthrough:**
1. Input/output are 2D, eps is scalar (`Tensor(0)`)
2. `tile((1, BLOCK_SIZE))` tiles along last dim only — `1` means "don't split dim 0", so each program processes one row
3. `eps` is returned as-is from arrangement (scalar, no tiling)
4. `ntl.sum(input_fp32 * input_fp32)` reduces over the block (row)
5. `input.shape[-1]` gives the block size for normalization

---

## Example 5: Matrix Multiplication (2D Tiling + Reduction)

**Input**: C = A @ B where A is (M, K), B is (K, N)

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()
BLOCK_SIZE_K = block_size()


def arrangement(
    input,
    other,
    output,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    output_arranged = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    input_arranged = input.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
    input_arranged = input_arranged.tile((1, -1))          # Split K into blocks of K'
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))  # Broadcast to N
    input_arranged.dtype = input_arranged.dtype.squeeze(0)  # Remove broadcast dim

    other_arranged = other.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
    other_arranged = other_arranged.tile((-1, 1))           # Split K into blocks of K'
    other_arranged = other_arranged.expand((output_arranged.shape[0], -1))  # Broadcast to M
    other_arranged.dtype = other_arranged.dtype.squeeze(1)  # Remove broadcast dim

    return input_arranged, other_arranged, output_arranged


def application(input, other, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(input.shape[0]):
        accumulator += ntl.dot(input[k], other[k])

    output = accumulator


tensors = (Tensor(2), Tensor(2), Tensor(2))
kernel = ninetoothed.make(arrangement, application, tensors)
```

**Walkthrough — the arrangement is the key insight:**

1. **Output**: Tiled as (BLOCK_SIZE_M, BLOCK_SIZE_N) — this is the outermost grid

2. **Input (A)**:
   - First tile: (BLOCK_SIZE_M, BLOCK_SIZE_K) — split M and K
   - Second tile: (1, -1) — further split K into blocks of size BLOCK_SIZE_K (creates 3 levels)
   - Expand: broadcast the inner dimension from K' to N (needed for dot product)
   - Squeeze dtype: remove the singleton broadcast dim from the dtype hierarchy

3. **Other (B)**:
   - First tile: (BLOCK_SIZE_K, BLOCK_SIZE_N) — split K and N
   - Second tile: (-1, 1) — further split K into blocks of size BLOCK_SIZE_K
   - Expand: broadcast from M' to M
   - Squeeze dtype: remove the singleton broadcast dim

4. **Application**: Loop over K blocks, accumulate dot products in fp32

The hierarchy for input becomes: `(M/M', K/K') -> (1, K') -> (M', N)` after expand+squeeze.

---

## Example 6: Softmax (Row Reduction with Numerical Stability)

**Input**: Softmax along last dimension of 2D tensor.

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((1, BLOCK_SIZE)), output.tile((1, BLOCK_SIZE))


def application(input, output):
    input_loaded = input

    row_minus_max = input_loaded - ntl.max(input_loaded)
    numerator = ntl.exp(row_minus_max)
    denominator = ntl.sum(numerator)

    output = numerator / denominator  # noqa: F841


tensors = (Tensor(2, other=float("-inf")), Tensor(2))
kernel = ninetoothed.make(arrangement, application, tensors)
```

**Walkthrough:**
1. `tile((1, BLOCK_SIZE))` — each program handles one full row (dim 0 is not split)
2. `other=float("-inf")` — out-of-bounds values (padding) are filled with -inf so they don't affect max
3. Subtract max before exp for numerical stability (prevents overflow)
4. `ntl.max` and `ntl.sum` reduce over the entire block (the row)

---

## Example 7: BMM — Batched Matrix Multiplication

**Input**: Batched matmul C[b] = A[b] @ B[b], where inputs are 3D.

```python
import ninetoothed
from ninetoothed import Tensor, block_size

from ops.ninetoothed.kernels.mm import application

BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()
BLOCK_SIZE_K = block_size()


def arrangement(
    input,
    other,
    output,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    output_arranged = output.tile((1, BLOCK_SIZE_M, BLOCK_SIZE_N))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    input_arranged = input.tile((1, BLOCK_SIZE_M, BLOCK_SIZE_K))
    input_arranged = input_arranged.tile((1, 1, -1))
    input_arranged = input_arranged.expand((-1, -1, output_arranged.shape[-1]))
    input_arranged.dtype = input_arranged.dtype.squeeze((0, 1))
    input_arranged.dtype.dtype = input_arranged.dtype.dtype.squeeze(0)

    other_arranged = other.tile((1, BLOCK_SIZE_K, BLOCK_SIZE_N))
    other_arranged = other_arranged.tile((1, -1, 1))
    other_arranged = other_arranged.expand((-1, output_arranged.shape[-2], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze((0, 2))
    other_arranged.dtype.dtype = other_arranged.dtype.dtype.squeeze(0)

    return input_arranged, other_arranged, output_arranged


tensors = (Tensor(3), Tensor(3), Tensor(3))
kernel = ninetoothed.make(arrangement, application, tensors)
```

**Walkthrough:**
1. Adds a batch dimension (`1`) as the outermost tile dimension — batch size becomes the grid size
2. Reuses `mm.application` directly — the inner computation is identical to non-batched matmul
3. Extra `.squeeze()` calls needed because the batch dim adds another level to the dtype hierarchy

---

## Example 8: Max Pool 2D (Sliding Window Reduction)

**Input**: Max pooling with configurable kernel size.

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)
WINDOW_HEIGHT = Symbol("WINDOW_HEIGHT", constexpr=True, upper_bound=16)
WINDOW_WIDTH = Symbol("WINDOW_WIDTH", constexpr=True, upper_bound=16)


def arrangement(input, output):
    input_arranged = input.tile((1, 1, WINDOW_HEIGHT, WINDOW_WIDTH))
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=4).flatten(start_dim=1)
    input_arranged = input_arranged.tile((BLOCK_SIZE, -1))

    output_arranged = output.tile((1, 1, 1, 1))
    output_arranged = output_arranged.ravel()
    output_arranged = output_arranged.flatten(end_dim=4).flatten(start_dim=1)
    output_arranged = output_arranged.tile((BLOCK_SIZE, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)

    return input_arranged, output_arranged


def application(input, output):
    output = ntl.max(input, axis=1)  # noqa: F841


kernel = ninetoothed.make(
    arrangement, application, (Tensor(4, other=float("-inf")), Tensor(4))
)
```

**Walkthrough:**
1. Tile input with window shape — creates (N, C, H_out, W_out) outer and (window_h, window_w) inner
2. `ravel()` flattens hierarchy into single level: (N, C, H_out, W_out, window_h, window_w)
3. `flatten(end_dim=4).flatten(start_dim=1)` merges dims into (N, C*H_out*W_out, window_h*window_w)
4. Second tile: (BLOCK_SIZE, -1) groups multiple output positions per program
5. `ntl.max(input, axis=1)` reduces over the window elements

---

## Example 9: Scaled Dot-Product Attention (Flash Attention)

**Input**: Multi-head attention with causal masking.

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()


def arrangement(
    q, k, v, scale, o, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
):
    def arrange_q_or_o(input):
        arranged = input.tile((1, 1, BLOCK_SIZE_M, -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1))
        return arranged

    def arrange_k_or_v(input):
        arranged = input.tile((1, 1, BLOCK_SIZE_N, -1))
        arranged = arranged.tile((1, 1, -1, -1))
        arranged = arranged.expand((-1, -1, q_arranged.shape[-2], -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
        return arranged

    q_arranged = arrange_q_or_o(q)
    return q_arranged, arrange_k_or_v(k), arrange_k_or_v(v), scale, arrange_q_or_o(o)


def application(q, k, v, scale, o):
    q_loaded = (q * scale * 1.44269504089).to(q.dtype)

    acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
    l_i = ntl.full((q.shape[-2],), 1, dtype=ntl.float32)
    m_i = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)

    for i in range(k.shape[0]):
        qk = ntl.dot(q_loaded, ntl.trans(k[i]))
        qk = ntl.where(k[i].offsets(-2) < k.source.shape[-2], qk, float("-inf"))

        m_ij = ntl.maximum(m_i, ntl.max(qk, 1))
        p = ntl.exp2(qk - m_ij[:, None])
        l_ij = ntl.sum(p, 1)

        alpha = ntl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + ntl.dot(p.to(v.dtype.dtype), v[i])
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    acc /= l_i[:, None]
    o = acc.to(o.dtype)  # noqa: F841


shape_options = (None, None, None, {"constexpr": True, "upper_bound": 128})
q, k, v, o = (Tensor(4, shape_options=shape_options) for _ in range(4))
tensors = (q, k, v, Tensor(0), o)
kernel = ninetoothed.make(arrangement, application, tensors)
```

**Walkthrough:**

1. **Q/O arrangement**: Tile head_dim with BLOCK_SIZE_M, keep batch+seq as grid
2. **K/V arrangement**: Tile head_dim with BLOCK_SIZE_N, add extra tile level for broadcasting to M, expand to match Q's grid, squeeze broadcast dims
3. **Scale**: Scalar constant, passed through as-is
4. **Application**: Online softmax with exp2 for efficiency
   - `1.44269504089` = log2(e) for the exp2 scaling trick
   - `k[i].offsets(-2) < k.source.shape[-2]` creates causal mask
   - `v.dtype.dtype` navigates the multi-level dtype hierarchy to get element type
5. **shape_options**: Head dimension is constexpr with upper bound 128 (common for LLMs)
