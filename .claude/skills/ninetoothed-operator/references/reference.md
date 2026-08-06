# Ninetoothed Operator Reference

## Repository Structure Analysis

### ninetoothed (Core Framework)

```
src/ninetoothed/
  __init__.py       # Public API: make, jit, build, Symbol, Tensor, block_size
  tensor.py         # Tensor class with meta-operations (tile, expand, squeeze, ...)
  symbol.py         # Symbol class — symbolic names, constexpr, meta, block_size()
  make.py           # make(arrangement, application, tensors) — integrates everything
  generation.py     # CodeGenerator — AST transformer, emits Triton code
  language.py       # Trampoline to triton.language
  jit.py            # JIT compilation (caller="torch")
  aot.py            # AOT compilation (caller="cuda")
  build.py          # Multi-config AOT build with auto-tuning
  torchifier.py     # AST visitor: ninetoothed names -> torch attribute accesses
  cudaifier.py      # AST visitor: ninetoothed names -> C-style struct accesses
```

**Key design:** `make()` calls `arrangement()` to get arranged tensors, attaches them as type annotations on `application()`, then `CodeGenerator` walks `application()`'s AST to generate Triton kernel code.

### ninetoothed-examples (Example Operators)

```
ops/ninetoothed/kernels/
  add.py, mm.py, bmm.py, addmm.py, softmax.py, silu.py,
  swiglu.py, rms_norm.py, fused_rms_norm.py, conv2d.py,
  max_pool2d.py, scaled_dot_product_attention.py,
  rotary_position_embedding.py
```

**Patterns observed:**
- Simple operators: `arrangement` returns directly tiled tensors
- Complex operators: `arrangement` chains multiple meta-operations (tile, expand, squeeze, ravel, flatten, permute)
- Conv2d decomposes into matmul via im2col arrangement
- BMM adds batch dimension handling on top of MM arrangement

### ntops (Production Operators)

```
src/ntops/
  kernels/           # Kernel implementations
    element_wise.py  # Shared arrangement for all element-wise ops
    reduction.py     # Shared arrangement for all reduction ops
    pooling.py       # Shared arrangement for pooling ops
    mm.py, conv2d.py, softmax.py, gelu.py, layer_norm.py, ...
  torch/             # Torch layer wrappers
    utils.py         # _cached_make, config defaults
  tests/             # Pytest-based test suite
```

**Key patterns:**
- **Shared arrangements**: `element_wise.arrangement`, `reduction.arrangement`, `pooling.arrangement` are reused across operators
- **`premake` pattern**: Functions that return `(arrangement, application, tensors)` tuple, allowing `functools.partial` for configuration
- **`_cached_make`**: `functools.cache`-wrapped `ninetoothed.make()` for kernel caching

## How `make()` Works Internally

```python
# ninetoothed/make.py (simplified)
def make(arrangement, application, tensors, **kwargs):
    params = inspect.signature(application).parameters
    types = arrangement(*tensors)           # Call arrangement with Tensor objects
    types = types if isinstance(types, tuple) else (types,)
    annotations = {param: type for param, type in zip(params, types)}
    application.__annotations__ = annotations  # Attach as type annotations
    # CodeGenerator then walks application's AST using these annotations
    return kernel  # JIT or AOT compiled
```

1. `arrangement` receives `Tensor` objects and returns arranged `Tensor` objects
2. Each returned tensor becomes the type annotation for the corresponding `application` parameter
3. `CodeGenerator` uses these annotations to generate pointer arithmetic, masks, and loads/stores
4. The outermost level of each arranged tensor defines the GPU launch grid

## Tensor Meta-Operations — Full Reference

### `tile(tile_shape, strides=None, dilation=None, floor_mode=False)`

Creates a 2-level hierarchical tensor. This is the **primary** operation for defining how tensors are split.

- `tile_shape`: Shape of each tile. `-1` means "use source dimension size"
- `strides`: Step between tiles. `-1` (default) = use tile size (non-overlapping)
- `dilation`: Spacing between elements within a tile. Default `[1, ...]`
- `floor_mode`: If True, uses floor division for outer shape computation

```python
# Split 1D tensor into blocks of BLOCK_SIZE
x.tile((BLOCK_SIZE,))
# Result: outer shape = (ceil(N/BLOCK_SIZE),), inner shape = (BLOCK_SIZE,)

# Split 2D tensor into BLOCK_SIZE_M x BLOCK_SIZE_K tiles
x.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))

# Sliding window (e.g., for conv2d im2col)
x.tile((1, kernel_h, kernel_w), strides=(-1, stride_h, stride_w), dilation=(1, 1, dilation_h, dilation_w))
```

### `expand(shape)`

Expands singleton dimensions. `-1` keeps original size.

```python
# Expand a (1, K) tensor to (M, K)
x.expand((M, -1))
```

### `squeeze(dim)`

Removes singleton dimensions. `dim` can be int or tuple.

```python
# Remove dim 0 from dtype hierarchy
x.dtype = x.dtype.squeeze(0)
```

### `unsqueeze(dim)`

Inserts a singleton dimension.

### `permute(dims)`

Reorders dimensions.

```python
# Transpose: (H, W) -> (W, H)
x.permute((1, 0))
```

### `flatten(start_dim=None, end_dim=None)`

Flattens a range of dimensions at one level. Default: flatten all.

### `ravel()`

Flattens the **entire hierarchy** into a single-level tensor. Different from `flatten` which only works within one level.

```python
# After tile((1, H, W)):
#   Level 0: (N,), Level 1: (1, H, W)
# After ravel():
#   Level 0: (N, 1, H, W)
```

### `pad(pad)`

Adds padding. `pad` is a list of `(left, right)` tuples per dimension.

```python
x.pad(((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
```

## Symbol System

### `Symbol(name, constexpr=True/False, meta=True/False, upper_bound=None)`

- `constexpr=True`: User-specified compile-time constant. Value provided at kernel call time.
- `meta=True`: Auto-tuned by the compiler. Multiple configs are benchmarked.
- `upper_bound`: For constexpr symbols, declares maximum value (enables compiler optimizations).
- No flags: Regular symbolic name (runtime value).

### `block_size()`

Shorthand for `Symbol("BLOCK_SIZE", meta=True)`. The compiler auto-tunes over power-of-2 values in [32, 1024]. **Prefer this over `Symbol(..., constexpr=True)`** for tile dimensions that should be tuned.

## Application Function Details

### Reading Tensors

```python
# Load entire block (2D case)
x  # loads the full block, returns a Triton tensor

# Load sub-block (3+ level hierarchy)
x[k]  # loads k-th sub-block

# Access shape info
x.shape[0]  # symbolic size of dimension 0
x.shape[-1]  # last dimension (supports negative indexing)
```

### Writing Tensors

```python
# Write to output block
output = result  # noqa: F841

# Write to sub-block (for in-place updates)
output[i] = result
```

### Loops and Reductions

```python
# Loop over reduction dimension
for k in range(input.shape[0]):
    accumulator += ntl.dot(input[k], other[k])

# Triton built-in reductions
ntl.sum(x)         # sum all elements
ntl.sum(x, axis=1) # sum along axis 1
ntl.max(x, axis=1) # max along axis 1
```

### Type Handling

```python
# Always upcast before math for fp16 inputs
x_fp32 = ntl.cast(x, ntl.float32)
result = ntl.exp(x_fp32)
result = ntl.cast(result, ntl.float16)

# Or use method syntax
result = x.to(ntl.float32)
```

## premake Pattern (ntops convention)

The `premake` function pattern separates configuration from kernel creation:

```python
import functools
from ninetoothed import Tensor

def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
    return arrangement_, application, tensors

# Usage
arr, app, tens = premake(ndim=2, dtype=ninetoothed.float16)
kernel = ninetoothed.make(arr, app, tens)
```

## Conv2d as Matmul Decomposition

Conv2d uses im2col to transform the convolution into a matrix multiplication:

1. **im2col**: Tile input with kernel-shaped windows, ravel into 2D matrix
2. **Reshape**: Flatten kernel and transpose
3. **Reuse**: Use mm.arrangement for the actual tiling strategy
4. **Application**: Can reuse mm.application directly, or add bias

```python
# arrangement (simplified)
input_arranged = input.tile((1, *kernel.shape[1:]), strides=(-1, -1, stride_h, stride_w))
input_arranged = input_arranged.squeeze(1)
input_arranged.dtype = input_arranged.dtype.squeeze(0)
input_arranged = input_arranged.ravel()
input_arranged = input_arranged.flatten(end_dim=3).flatten(start_dim=1)

filter_arranged = filter.flatten(start_dim=1).permute((1, 0))
output_arranged = output.permute((0, 2, 3, 1)).flatten(end_dim=3)

return mm.arrangement(input_arranged, filter_arranged, output_arranged)
```

## Advanced: Flash Attention Pattern

Online softmax algorithm that avoids materializing the full attention matrix:

```python
def application(q, k, v, scale, o):
    q_loaded = (q * scale * 1.44269504089).to(q.dtype)  # log2(e) scaling
    acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
    l_i = ntl.full((q.shape[-2],), 1, dtype=ntl.float32)  # running sum
    m_i = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)  # running max

    for i in range(k.shape[0]):
        qk = ntl.dot(q_loaded, ntl.trans(k[i]))
        qk = ntl.where(k[i].offsets(-2) < k.source.shape[-2], qk, float("-inf"))

        m_ij = ntl.maximum(m_i, ntl.max(qk, 1))
        p = ntl.exp2(qk - m_ij[:, None])
        l_ij = ntl.sum(p, 1)

        alpha = ntl.exp2(m_i - m_ij)  # correction factor
        acc = acc * alpha[:, None] + ntl.dot(p.to(v.dtype.dtype), v[i])
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    acc /= l_i[:, None]
    o = acc.to(o.dtype)  # noqa: F841
```

**Key techniques:**
- `exp2` + `log2(e)` scaling instead of `exp` for efficiency
- Online softmax: track running max (`m_i`) and sum (`l_i`), apply correction factor (`alpha`)
- `k[i].offsets(-2) < k.source.shape[-2]` for causal masking
- Multi-level dtype hierarchy: `v.dtype.dtype` to access the element dtype through multiple levels

## Rotary Position Embedding (RoPE)

Uses dilation in `tile()` to select alternating elements:

```python
def arrangement(input, sin_table, cos_table, interleaved=True):
    tile_shape = (1, 1, 1, emb_dim // 2)
    if interleaved:
        strides = (-1, -1, -1, 1)
        dilation = (1, 1, 1, 2)  # stride-2 to select every other element
    else:
        strides = None
        dilation = None

    input_arranged = input.tile(tile_shape, strides=strides, dilation=dilation)
    input_arranged = input_arranged.tile((1, 1, 1, 2))  # duplicate for sin/cos pair
    # ... squeeze dtype hierarchy ...
```

**Key technique:** `dilation=2` in tile selects every other element, creating the sin/cos pairs for rotation.
