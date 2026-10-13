# NineToothed DSL Quick Reference

Use this reference when choosing DSL patterns for an `ntops` operator.

Source: distilled from the checked-in `ninetoothed/src/ninetoothed/`, `ntops/src/ntops/`, and their tests. It is a repository-oriented quick reference, not a substitute for the upstream API documentation.

## Common Building Blocks

- `Tensor(ndim, dtype=None)`: describes an input/output tensor surface for `ninetoothed.make`.
- `block_size()`: declares a tunable block dimension.
- `arrangement`: maps logical tensors into tiled, expanded, flattened, or otherwise arranged views.
- `application`: expresses elementwise, reduction, or matrix computation on arranged tensors.
- `premake(...)`: returns `(arrangement, application, tensors)` for wrapper-side cached compilation.

## Elementwise Pattern

For simple unary or binary elementwise operators, prefer the shared `ntops.kernels.element_wise.arrangement` helper. The operator-specific file usually only needs:

- imports,
- an `application(input, output)` function,
- a `premake(ndim, dtype=None, block_size=None)` function,
- `Tensor` descriptors for inputs and outputs.

## Wrapper Pattern

Torch wrappers usually:

- allocate `output = torch.empty_like(input)` or a shape-specific tensor,
- build a cached kernel with `_cached_make(...)`,
- invoke `kernel(input, output, ...)`,
- return the output tensor.

## Risk Points

- Broadcasting semantics must match PyTorch.
- Non-contiguous tensors require explicit stride/layout attention.
- Float16 comparisons need looser tolerances than float32.
- Some generated kernels require CUDA even when Python import succeeds.
- Approximation modes, such as GELU `approximate="tanh"`, should be documented if incomplete.
- Treat `expand` on inputs and outputs differently. An expanded input is a read-only broadcast view that still needs stride verification; writing distinct reduction results through a zero-stride expanded output aliases addresses and can break pointer/mask lowering.
- Minimize simultaneous symbolic transforms. Establish a working fixed-rank load/reduce/store first, then add `permute`, `unsqueeze`, `expand`, or dynamic tiling one at a time with a fresh CUDA check.
- A generated-source cache hit is evidence only after linking it to the current operator invocation. Low/zero operator-match results from `--no-trigger` can be unrelated historical kernels.
