# NineToothed API Notes

Use this reference when a task depends on NineToothed-specific tensor meta-programming. These notes distill the official docs and public repository patterns into checks an AI agent should perform before editing code.

## Mental Model

NineToothed uses symbolic tensors, not real data tensors, when defining kernels. A `Tensor` stores symbolic shape, strides, dtype nesting, optional out-of-bound fill values, and a history of meta-operations.

Operator development usually follows:

```text
source Tensor specs -> arrangement meta-operations -> aligned outer launch shape -> application over per-program blocks -> ninetoothed.make or @ninetoothed.jit
```

The most important rule: after arrangement, the outermost tensors of all non-scalar parameters must line up so the compiler can launch programs over a shared grid. The application function receives the per-program block values, not the original source tensors.

## Symbols

Use:

```python
from ninetoothed import Symbol, block_size

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
AUTO_BLOCK = block_size()
```

Guidance:

- Use `block_size()` when matching existing auto-tuned examples.
- Use `Symbol(..., constexpr=True)` for values provided at launch or compile time.
- Use `Symbol(..., meta=True)` when the repository uses meta parameters for tuning.
- Keep symbol names consistent with local style, e.g. `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`.

## Tensor Construction

Common forms:

```python
Tensor(1)
Tensor(2)
Tensor(0)
Tensor(2, other=float("-inf"))
Tensor(shape=(m, n))
Tensor(4, shape_options={"constexpr": True})
```

Guidance:

- `Tensor(rank)` describes runtime tensor rank.
- `Tensor(0)` is used for scalar arguments such as `alpha` or `beta`.
- `other` supplies out-of-bounds values for tail tiles; choose identity carefully.
- `shape_options={"constexpr": True}` is useful in debug/eval contexts or examples with fixed shape metadata.

## Arrangement Operations

Useful meta-operations:

- `tile(tile_shape, strides=None, dilation=None, floor_mode=False)`
- `expand(shape)`
- `squeeze(dim_or_dims)`
- `unsqueeze(dim)`
- `permute(order)`
- `flatten(start_dim=..., end_dim=...)`
- `ravel()`
- `pad(...)`

Checklist:

- After `tile`, remember the tensor has an outer shape and an inner dtype shape.
- Use `expand` to align outer launch shapes, as in matrix multiplication.
- Use `dtype.squeeze(...)` when the inner dtype level contains a useless singleton dimension.
- Use `flatten`, `ravel`, and `permute` to map high-rank operators onto known lower-rank kernels only when a nearby example uses the same style.
- Avoid changing layout semantics accidentally while flattening or permuting.

## Application Operations

Use `ninetoothed.language as ntl` for kernel expressions:

```python
import ninetoothed.language as ntl

accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
accumulator += ntl.dot(input[k], other[k])
row_minus_max = input - ntl.max(input)
numerator = ntl.exp(row_minus_max)
output = numerator / ntl.sum(numerator)
```

Guidance:

- Accumulate fp16 matrix/reduction intermediates in `ntl.float32` unless local examples do otherwise.
- Use stable formulas for softmax and normalization.
- Treat `output = ...  # noqa: F841` as the NineToothed assignment idiom when local code uses it.

## Debugging Arrangement

For complex arrangements, inspect mapping before runtime debugging:

```python
from ninetoothed.debugging import simulate_arrangement

source_tensors, target_tensors = simulate_arrangement(arrangement, tensors)
```

Note: `simulate_arrangement` currently supports CUDA only (see `tests/test_debugging.py`). On non-CUDA machines, fall back to the symbolic `eval` check below.

Alternative for concrete symbolic tensors:

```python
for arranged in arrangement(*tensors):
    if arranged.ndim != 0:
        print(arranged.flatten().eval())
```

Use these checks when:

- the task uses `tile` more than once
- `expand` aligns matrix-like operands
- `flatten`, `ravel`, or `permute` changes rank
- tail tiles require `other`
- tests fail with wrong indexing or wrong boundary values

## AOT and Generated Source

In the core repository, AOT is triggered through `ninetoothed.make(arrangement, application, tensors, caller="cuda", kernel_name=..., output_dir=...)`; `caller="torch"` (the default) goes through JIT instead. There is no separate public AOT entry point for operator tasks; `tests/test_aot.py` uses `make(..., caller=device, output_dir=ninetoothed.generation.CACHE_DIR)`.

Key facts:

- The AOT path compiles generated C sources with `nvcc -shared` (see `src/ninetoothed/aot.py`). PyTorch CUDA runtime alone is not enough; a full CUDA Toolkit with `nvcc` on PATH is required. Verify with `nvcc --version` before diagnosing AOT failures as code bugs.
- `ninetoothed.build(premake, configs, ...)` is a separate ahead-of-time multi-config build API (see `src/ninetoothed/build.py` and `docs/source/build.rst`); `configs` belongs to `build`, not `make`.
- Keep `kernel_name`, `output_dir`, caller, `num_warps`, and `num_stages` explicit.
- Inspect generated dispatcher variants for divisibility and contiguity assumptions.
- Record exact output files and build command.

## Frequent Mistakes

- Arranged tensors have incompatible outer shapes.
- Application indexes the wrong tensor level.
- Tail tiles use the wrong identity fill.
- Scalar `Tensor(0)` parameters are tiled like data tensors.
- A layout-sensitive task silently calls `.contiguous()`.
- Tests compare against a contiguous clone instead of the original view.
- Benchmark uses different shapes or dtype from the correctness test without saying so.
- Passing a tensor whose ndim differs from the declared `Tensor(rank)`: the generated launch reads only `size(0..rank-1)`/`stride(0..rank-1)` and does NOT raise, silently processing a fraction of the elements and leaving the rest of the output uninitialized. Always check that argument rank matches the kernel declaration; generated kernels are stride-aware only when ranks match.
- Allocating output with `torch.empty_like(non_dense_view)`: it returns a contiguous tensor, so writes through view-based stride assumptions land in the wrong places or leave gaps.
- Referencing module-level Python constants inside `application`: the application source is extracted and compiled standalone, so globals are NOT captured and fail with `NameError` at Triton compile time. Inline numeric constants in the application body, or pass them as `Tensor(0)` scalar parameters.
- Needing math functions not in `ntl` (e.g. `tanh`): use `from ninetoothed.language import libdevice` (re-export of `triton.language.extra.libdevice`), as core tests do with `libdevice.sin`/`libdevice.pow`.
