# DSL Pattern Index

This is an index of patterns to fill as the project evolves. Keep examples
short and point to live repository files.

## Elementwise Pattern

- repository anchors: `tests/test_add.py`, `tests/test_pow.py`,
  `tests/test_dropout.py`, `tests/test_clone.py`, `src/ninetoothed/make.py`.
- common arrangement shape: one or more `Tensor(1).tile((BLOCK_SIZE,))`
  views, usually paired with a single output tile.
- common application shape: direct elementwise expressions in `@ninetoothed.jit`
  or `make(..., application, ...)` with no explicit loop.
- tests to mirror: start with `test_add.py`; use `test_pow.py` for scalar-like
  parameters and `test_clone.py` for copy-style application forms.
- known traps: forgetting `constexpr` on scalar block sizes, assuming
  contiguous-only inputs when the task allows views, and changing output dtype
  metadata after tiling.

## Broadcast Pattern

- repository anchors: `tests/test_expand.py`, `tests/test_unsqueeze.py`,
  `tests/test_generation.py`, `tests/test_attention.py`,
  `src/ninetoothed/tensor.py`.
- shape extraction: use `Tensor.shape`, `shape_options`, `expand`, `unsqueeze`,
  `squeeze`, `flatten`, and `permute` in the arrangement, then keep the
  application compact.
- dtype handling: align tensor metadata after shape-changing meta-operations;
  see the `dtype.squeeze(...)` adjustments in `test_generation.py` and
  `test_attention.py`.
- mask handling: use `other=` sentinels or `ntl.where(...)` when broadcasted
  values may fall outside a valid tile or causal window.
- known traps: expanding the wrong axis order, forgetting that broadcasted
  tensors still need explicit output shape metadata, and relying on PyTorch
  broadcasting where the arrangement must spell it out.

## Reduction Pattern

- repository anchors: `tests/test_softmax.py`, `tests/test_max_pool2d.py`,
  `tests/test_matmul.py`, `tests/test_attention.py`.
- tile/block decision: start from `block_size()` or explicit `BLOCK_SIZE_*`
  symbols, then pick tile shapes that line up with the reduced axis and the
  output layout.
- numerical stability: prefer `ntl.max` + shift + `ntl.exp` + `ntl.sum` for
  softmax-like paths; use running `max`/normalization patterns when the loop is
  blocked.
- tail block handling: pair `other=float("-inf")`, `floor_mode`, or explicit
  offset guards with the reduction window so out-of-range lanes do not leak.
- known traps: reducing before the window is fully masked, forgetting the
  output accumulator dtype, and assuming one block shape fits all input sizes.

## Layout-Sensitive Pattern

- repository anchors: `tests/test_clone.py`, `tests/test_conv2d.py`,
  `tests/test_jagged.py`, `tests/test_eval.py`,
  `src/ninetoothed/tensor.py`, `src/ninetoothed/generation.py`.
- stride/offset handling: use `tensor.offsets(...)`, `tensor.source.stride(...)`,
  `data_ptr()`, or explicit `tile(..., strides=...)` forms when layout matters.
- non-contiguous tests: mirror `test_clone.py` for strided views,
  `test_conv2d.py` for tiled transforms with stride-aware arrangement, and
  `test_jagged.py` for jagged metadata.
- fallback strategy: if a layout-sensitive path is too large, preserve
  correctness with a contiguous fallback or a clearly documented unsupported
  scope entry.
- known traps: assuming `.shape` implies contiguous storage, forgetting
  `offsets()` in generated indexing, and collapsing layout metadata too early
  with `flatten()`/`ravel()`/`permute()`.

## AOT / Generated Source Pattern

- repository anchors: `tests/test_generation.py`, `tests/test_aot.py`,
  `src/ninetoothed/aot.py`, `src/ninetoothed/generation.py`,
  `src/ninetoothed/jit.py`.
- generated source path: inspect `kernel._source` for JIT output and
  `ninetoothed.generation.CACHE_DIR` for cached/generated files.
- AOT build command: the helper path in `src/ninetoothed/aot.py` ultimately
  shells out through `python -m triton.tools.compile` and `nvcc -shared -arch
  native`; mirror `tests/test_aot.py` for coverage.
- debug checklist: confirm `kernel_name`, `caller`, `output_dir`, emitted
  `.cpp`/`.h` files, variant dispatch, and whether contiguity/divisibility
  checks match the task.
- known traps: reading only the Python wrapper and missing the generated C/C++
  artifacts, assuming one AOT variant covers all layouts, and claiming
  performance wins without a reproducible build or source inspection.
