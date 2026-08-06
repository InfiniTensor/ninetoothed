# Operator Patterns

Use this reference when writing or modifying NineToothed operator code.

## Repository Recon

Run searches before editing:

```shell
rg -n "def arrangement|def application|ninetoothed.make|@ninetoothed.jit|Tensor\\(|Symbol\\(" .
rg -n "softmax|add|relu|gelu|max_pool|stride|offset|contiguous|benchmark|generated|aot" .
```

Prefer the closest existing operator by category:

- Elementwise/broadcast: `add`, `silu`, `swiglu`
- Reduction/block: `softmax`, `rms_norm`, `max_pool2d`
- Matrix/tile reuse: `mm`, `bmm`, `addmm`, `conv2d`
- Layout-sensitive examples: operators passing strides, offsets, reshapes, flatten/ravel, or non-contiguous tests

## Arrange-and-Apply Skeleton

The common module pattern is:

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    input_arranged = input.tile((BLOCK_SIZE,))
    output_arranged = output.tile((BLOCK_SIZE,))
    return input_arranged, output_arranged


def application(input, output):
    output = input  # noqa: F841


tensors = (Tensor(1), Tensor(1))
kernel = ninetoothed.make(arrangement, application, tensors)
```

Inline tests may instead use:

```python
@ninetoothed.jit
def op_kernel(input: Tensor(1).tile((BLOCK_SIZE,)), output: Tensor(1).tile((BLOCK_SIZE,))):
    output = input  # noqa: F841
```

Match the surrounding style.

Before writing `application`, verify the arrangement contract:

```text
source input shapes -> arranged outer shapes -> per-program dtype/block shapes
```

For non-scalar tensors, the arranged outer shapes should match. If they do not, add `expand`, `tile`, `flatten`, `ravel`, or `permute` only after checking a nearby example.

## Tensor and Symbol Choices

- Use `ninetoothed.block_size()` when matching auto-tuned examples.
- Use `Symbol("NAME", constexpr=True)` for compile-time parameters such as a fixed tile or reduction length.
- Use `Symbol("NAME", meta=True)` when the repository uses auto-tuned or meta scheduling parameters.
- Use `Tensor(rank)` for normal tensors.
- Use `Tensor(rank, other=<fill>)` for masked-out values in reductions, e.g. `float("-inf")` for max/softmax.
- Keep rank and tiling visible; avoid clever helper abstractions in contest tasks.

## Elementwise Operators

Checklist:

- Tile all input/output tensors on the same logical element block.
- Preserve dtype unless the task asks for promotion.
- Cover broadcast explicitly; do not silently assume equal shapes.
- Store only once to the output.

Typical pattern:

```python
def arrangement(input, other, output, BLOCK_SIZE=BLOCK_SIZE):
    return (
        input.tile((BLOCK_SIZE,)),
        other.tile((BLOCK_SIZE,)),
        output.tile((BLOCK_SIZE,)),
    )


def application(input, other, output):
    output = input + other  # noqa: F841
```

## Reduction and Block Operators

Checklist:

- Make the reduction axis clear.
- Use stable formulas when needed, e.g. subtract max before softmax exp.
- Use `other=float("-inf")` or another identity fill when tiles can extend past valid data.
- Squeeze or adjust dtype shape when the output rank differs from the arranged input.
- For sum-like reductions, choose `0` as fill; for max-like reductions, choose `float("-inf")`; for min-like reductions, choose `float("inf")`.

Softmax-style pattern:

```python
def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((1, BLOCK_SIZE)), output.tile((1, BLOCK_SIZE))


def application(input, output):
    row_minus_max = input - ntl.max(input)
    numerator = ntl.exp(row_minus_max)
    output = numerator / ntl.sum(numerator)  # noqa: F841


tensors = (Tensor(2, other=float("-inf")), Tensor(2))
```

## Layout-Sensitive Operators

Never assume `.contiguous()` is acceptable unless the task allows a copy. For stride, offset, or non-contiguous work:

- First check rank: NineToothed generated kernels compute per-dimension stride-aware offsets, so a rank-matched kernel (`Tensor(2)` for a 2-D view) handles non-contiguous inputs natively. Most layout failures come from passing an N-D view to a lower-rank kernel, which silently reads only `size(0)`/`stride(0)` without raising.
- Inspect existing wrapper signatures for stride arguments and for `.flatten()` calls (flatten copies non-contiguous views, which is a documented-copy fix, not native support).
- Check whether `Tensor` metadata, arrangement transformations, or wrapper-level pointer/stride handling already solves the case.
- Beware `torch.empty_like(non_dense_view)`: it returns a contiguous tensor, which changes write-side layout assumptions.
- Add tests that create non-contiguous inputs using transpose, slicing with step, narrow, permute, or as_strided when safe.
- Compare against PyTorch on the exact same view, not a contiguous clone, unless the operator contract requires clone behavior.
- State unsupported layouts explicitly.

## Debugging Arrangement

Use this when shape reasoning is uncertain:

```python
from ninetoothed.debugging import simulate_arrangement

source_tensors, target_tensors = simulate_arrangement(arrangement, tensors)
```

If debugging extras are unavailable, instantiate concrete `Tensor(shape=...)` objects and inspect:

```python
for arranged in arrangement(*tensors):
    if arranged.ndim != 0:
        print(arranged.flatten().eval())
```

Do this before changing application logic when the failure looks like wrong block mapping or tail handling.

## Minimal Patch Standard

A good contest patch:

- adds only the operator, wrapper, test, benchmark, or docs requested
- follows local naming and import order
- avoids global formatting churn
- keeps tolerances justified by dtype and operation
- includes exact commands and results
