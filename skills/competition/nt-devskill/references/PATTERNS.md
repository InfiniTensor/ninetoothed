# Common NineToothed Operator Patterns

## 1. Element-wise (Vector Add / Mul / SiLU)

```python
def arrangement(x, y, out, BLOCK_SIZE=BLOCK_SIZE):
    return x.tile((BLOCK_SIZE,)), y.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(x, y, out):
    out = x + y  # or x * y, etc.
```

**Key points:** 1D tiling, no cross-element dependency, no reduction.

### SiLU / Activation Functions

```python
def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

def application(input, output):
    output = input * ntl.sigmoid(ntl.cast(input, ntl.float32))
```

**Key points:** Cast to float32 for sigmoid precision, then compute activation. Pattern applies to SwiGLU, GELU, etc.

---

## 2. Reduce (Softmax / Sum)

```python
def arrangement(x, out, BLOCK_SIZE=BLOCK_SIZE):
    return x.tile((1, BLOCK_SIZE)), out.tile((1, BLOCK_SIZE))

def application(x, out):
    row_minus_max = x - ntl.max(x)
    numerator = ntl.exp(row_minus_max)
    denominator = ntl.sum(numerator)
    out = numerator / denominator
```

**Key points:** 2D tiling with `(1, BLOCK_SIZE)` processes one row at a time. Use `Tensor(2, other=float("-inf"))` for the input to handle partial tiles correctly. The `other` value fills out-of-bounds positions during tiling.

---

## 3. Matrix Multiplication (MatMul / BMM)

```python
def arrangement(a, b, out, BM, BN, BK):
    out = out.tile((BM, BN))
    a = a.tile((BM, BK)).tile((1, -1)).expand((-1, out.shape[1]))
    a.dtype = a.dtype.squeeze(0)
    b = b.tile((BK, BN)).tile((-1, 1)).expand((out.shape[0], -1))
    b.dtype = b.dtype.squeeze(1)
    return a, b, out

def application(a, b, out):
    acc = ntl.zeros(out.shape, dtype=ntl.float32)
    for k in range(a.shape[0]):
        acc += ntl.dot(a[k], b[k])
    out = acc
```

**Key points:** 3 tile dimensions (M, N, K), K-loop with `ntl.dot`, accumulator in float32.

### Fusion: MatMul + Bias + ReLU

```python
def application(a, b, bias, out):
    acc = ntl.zeros(out.shape, dtype=ntl.float32)
    for k in range(a.shape[0]):
        acc += ntl.dot(a[k], b[k])
    out = ntl.maximum(acc + bias, 0)
```

---

## 4. Normalization (RMS Norm / LayerNorm)

```python
def arrangement(x, w, eps, y, BLOCK_SIZE=BLOCK_SIZE):
    def arrange(tensor):
        return tensor.tile((1, BLOCK_SIZE))
    return arrange(x), arrange(w), eps, arrange(y)

def application(x, w, eps, y):
    x_fp32 = ntl.cast(x, ntl.float32)
    y = x_fp32 * ntl.rsqrt(ntl.sum(x_fp32 * x_fp32) / x.shape[-1] + eps) * w
```

**Key points:** Cast to float32 for precision, `ntl.rsqrt` is more efficient than `1/ntl.sqrt()`. Both `x` and `w` are 2D (weight broadcast via tiling).

---

## 5. Attention (Scaled Dot-Product / Flash Attention)

```python
def application(q, k, v, scale, o):
    q = (q * scale * 1.44269504089).to(q.dtype)  # scale * log2(e)
    acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
    l_i = ntl.full((q.shape[-2],), 1, dtype=ntl.float32)
    m_i = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)

    for i in range(k.shape[0]):
        qk = ntl.dot(q, ntl.trans(k[i]))
        qk = ntl.where(k[i].offsets(-2) < k.source.shape[-2], qk, float("-inf"))

        m_ij = ntl.maximum(m_i, ntl.max(qk, 1))
        p = ntl.exp2(qk - m_ij[:, None])
        l_ij = ntl.sum(p, 1)

        alpha = ntl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + ntl.dot(p.to(v.dtype.dtype), v[i])
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    acc /= l_i[:, None]
    o = acc.to(o.dtype)
```

**Key points:** Online softmax (safe softmax), K/V tile loop, `ntl.where` for masking, `exp2` for optimization (convert `exp` to `exp2` via `* log2(e)`), `.offsets()` + `.source.shape` for bounds checking, `Tensor(4, shape_options=...)` for compile-time constraints.

---

## 6. Kernel Composition (Reuse Pattern)

NineToothed supports reusing `arrangement` and `application` functions across kernels. This is a key architectural advantage over Triton.

### Batched MatMul (reusing MatMul's application)

```python
from examples.matmul.kernel import application  # 直接复用 mm 的计算逻辑

def arrangement(input, other, output, BM=..., BN=..., BK=...):
    # 只写 3D tiling 的 arrangement
    output_arranged = output.tile((1, BM, BN))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)
    # ... (3D-specific tiling)
    return input_arranged, other_arranged, output_arranged

# 组合：自己的 arrangement + mm 的 application
kernel = ninetoothed.make(arrangement, application, tensors)
```

### AddMM: beta*C + alpha*A@B (reusing MatMul's arrangement + application)

```python
import examples.matmul.kernel as mm

def arrangement(input, mat1, mat2, beta, alpha, output):
    _, _, input_arranged = mm.arrangement(mat1, mat2, input)      # bias 视图
    mat1_arr, mat2_arr, output_arr = mm.arrangement(mat1, mat2, output)  # matmul 视图
    return input_arranged, mat1_arr, mat2_arr, beta, alpha, output_arr

def application(input, mat1, mat2, beta, alpha, output):
    mm.application(mat1, mat2, output)              # 先执行 matmul
    output = beta * input + alpha * output          # 再融合 bias

kernel = ninetoothed.make(arrangement, application, tensors)
```

**Composition guidelines:**
- 先检查已有 kernel 的 arrangement/application 是否可以复用
- `application` 函数签名必须匹配 `tensors` 中的参数顺序
- 组合时注意 `dtype.dtype` 多层访问（3D+ tiling 时需要）

---

## 7. Convolution (im2col + MatMul)

```python
import examples.matmul.kernel as mm

def arrangement(input, filter, output):
    input_arranged = input.tile((1, *filter.shape[1:]), strides=(-1, -1, 1, 1))
    input_arranged = input_arranged.squeeze(1)
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=3).flatten(start_dim=1)

    filter_arranged = filter.flatten(start_dim=1)
    filter_arranged = filter_arranged.permute((1, 0))

    output_arranged = output.permute((0, 2, 3, 1)).flatten(end_dim=3)

    return mm.arrangement(input_arranged, filter_arranged, output_arranged)

# 完全复用 mm 的 arrangement 和 application
kernel = ninetoothed.make(arrangement, mm.application, tensors)
```

**Key points:** im2col via tiling + squeeze + ravel + flatten, then delegate to matmul.

---

## 8. Gated Activation (SwiGLU)

```python
def arrangement(a, b, c, BLOCK_SIZE=BLOCK_SIZE):
    return a.tile((BLOCK_SIZE,)), b.tile((BLOCK_SIZE,)), c.tile((BLOCK_SIZE,))

def application(a, b, c):
    gate = b * ntl.sigmoid(ntl.cast(b, ntl.float32))
    c = a * gate
```

**Key points:** Two-input gated pattern. Cast `b` to float32 for sigmoid precision. The output `c = a * (b * sigmoid(b))` is the SwiGLU activation. Same 1D tile pattern as SiLU but with two inputs.

---

## 9. Pooling (MaxPool2D)

```python
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
    output = ntl.max(input, axis=1)

kernel = ninetoothed.make(arrangement, application, (Tensor(4, other=float("-inf")), Tensor(4)))
```

**Key points:** Window-based tiling with constexpr window dimensions. `Tensor(other=float("-inf"))` ensures partial windows produce correct max results. `ntl.max(axis=1)` reduces across the window.

---

## 10. Rotary Position Embedding (RoPE)

```python
import functools

def arrangement(input, sin_table, cos_table, interleaved=True):
    emb_dim = input.shape[-1]
    tile_shape = (1, 1, 1, emb_dim // 2)

    if interleaved:
        strides = (-1, -1, -1, 1)
        dilation = (1, 1, 1, 2)
    else:
        strides = None
        dilation = None

    input_arranged = input.tile(tile_shape, strides=strides, dilation=dilation)
    input_arranged = input_arranged.tile((1, 1, 1, 2))
    input_arranged.dtype = input_arranged.dtype.squeeze((0, 1, 2))
    input_arranged.dtype.dtype = input_arranged.dtype.dtype.squeeze((0, 1, 2))

    sin_table_arranged = sin_table.tile(tile_shape)
    sin_table_arranged.dtype = sin_table_arranged.dtype.squeeze((0, 1, 2))

    cos_table_arranged = cos_table.tile(tile_shape)
    cos_table_arranged.dtype = cos_table_arranged.dtype.squeeze((0, 1, 2))

    return input_arranged, sin_table_arranged, cos_table_arranged

def application(input, sin_table, cos_table):
    input_0 = input[0]
    input_1 = input[1]
    input[0] = input_0 * cos_table - input_1 * sin_table
    input[1] = input_0 * sin_table + input_1 * cos_table

# Use functools.partial for multiple variants
interleaved_kernel = ninetoothed.make(
    functools.partial(arrangement, interleaved=True), application, inputs
)
non_interleaved_kernel = ninetoothed.make(
    functools.partial(arrangement, interleaved=False), application, inputs
)
```

**Key points:** 4D tiling with `dilation` for interleaved access pattern. Uses `functools.partial` to create multiple kernel variants from the same arrangement. Requires double-level `dtype.dtype` access for 4D tiling. The rotation formula is `x_rot[0] = x[0]*cos - x[1]*sin`, `x_rot[1] = x[0]*sin + x[1]*cos`.

---

## Pattern Selection Guide

| Computation Pattern       | Tile Strategy                 | Key NTL Primitives                        |
| ------------------------- | ----------------------------- | ----------------------------------------- |
| Element-wise              | 1D tile                       | `+`, `*`, `ntl.maximum`                   |
| Activation (SiLU)         | 1D tile                       | `ntl.cast`, `ntl.sigmoid`                 |
| Gated Activation (SwiGLU) | 1D tile, 2 inputs             | `ntl.cast`, `ntl.sigmoid`                 |
| Reduce (row-wise)         | 2D tile `(1, BLOCK)`          | `ntl.max`, `ntl.sum`, `Tensor(other=...)` |
| MatMul                    | 3 tile dims, K-loop           | `ntl.dot`, float32 accumulator            |
| Normalization             | 2D tile + weight 2D tile      | `ntl.cast`, `ntl.rsqrt`, `ntl.sum`        |
| Attention                 | 4D tile, KV loop              | `ntl.dot`, `ntl.exp2`, online softmax     |
| Convolution               | im2col + MatMul               | `.ravel()`, `.flatten()`, `.permute()`    |
| Pooling (MaxPool2D)       | Window tile                   | `ntl.max`, `Tensor(other=-inf)`           |
| Rotary Embedding (RoPE)   | 4D tile + dilation            | `functools.partial`, `dtype.dtype`        |
| Composition               | Reuse arrangement/application | Import from existing kernels              |
