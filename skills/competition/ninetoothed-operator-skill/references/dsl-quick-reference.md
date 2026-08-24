# NineToothed DSL Quick Reference

## Import pattern
```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor, block_size
```

## Symbol types
| Declaration | Meaning |
|---|---|
| `Symbol("X", constexpr=True)` | Compile-time constant, must be provided at call time |
| `Symbol("X", meta=True)` | Autotuning symbol — NineToothed searches for optimal value |
| `block_size()` | Shorthand for `Symbol(..., meta=True)` for block sizes |
| `Symbol("X", constexpr=True, upper_bound=N)` | Bounded constexpr (e.g., window sizes) |

## Tensor declaration
| Declaration | Meaning |
|---|---|
| `Tensor(n)` | n-dimensional tensor |
| `Tensor(0)` | Scalar |
| `Tensor(n, other=float("-inf"))` | n-dim tensor; OOB tiles filled with -inf |

## Arrangement operations
| Operation | Effect |
|---|---|
| `t.tile((A, B))` | Partition tensor into (A, B)-shaped tiles |
| `t.tile((-1, B))` | -1 means "iterate over this dim", don't fix size |
| `t.tile((1, B))` | 1 means "process one element in this dim per program" |
| `t.ravel()` | Flatten all dims into 1D |
| `t.flatten(start_dim=i, end_dim=j)` | Flatten dims i..j |
| `t.expand((A, B))` | Broadcast-expand to match another arrangement shape |
| `t.dtype.squeeze(dim)` | Remove a tile dimension from the dtype |

## ntl operations
```python
ntl.max(x)              # reduce max over all elements of tile
ntl.max(x, axis=1)      # reduce max over axis 1
ntl.min(x)
ntl.sum(x)
ntl.sum(x, axis=0)
ntl.exp(x)
ntl.log(x)
ntl.sqrt(x)
ntl.rsqrt(x)            # 1/sqrt(x)
ntl.sigmoid(x)
ntl.cast(x, ntl.float32)  # dtype cast
ntl.zeros(shape, dtype=ntl.float32)
ntl.dot(a, b)           # matrix multiply tiles
```

## Kernel construction
```python
kernel = ninetoothed.make(arrangement, application, tensors)
```
- `tensors` must be a tuple of `Tensor(...)` matching `arrangement` signature order
- The kernel is callable: `kernel(x, y, output, BLOCK_SIZE=1024)`

## Debugging env vars
```bash
NINETOOTHED_DUMP_GENERATED_SOURCE=1  # print generated Triton source
```

## Common tile patterns summary
```
1D elementwise:     tile((BLOCK_SIZE,))
Row-wise reduction: tile((1, BLOCK_SIZE))  + Tensor(2, other=float("-inf"))
GEMM/matmul:        out of scope for this skill package
Spatial window:     ravel() + flatten() + tile((BLOCK_SIZE, -1))
```
