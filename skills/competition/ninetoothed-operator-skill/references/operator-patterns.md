# Operator Patterns

Use these patterns after `SKILL.md` Step 2 selects an implementation family.
These are templates, not proof that every pattern is validated by this package.

## Elementwise / Broadcast

Validated self-test: ReLU.

```python
import ninetoothed


def arrangement(*tensors, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()
    return tuple(
        t.flatten().tile((block_size,)) if t.ndim != 0 else t
        for t in tensors
    )


def application(input, output):
    output = max(0.0, input)  # noqa: F841
```

Use for add, mul, relu, gelu, silu, swiglu, and fused elementwise operators.
Test broadcast semantics explicitly; do not assume a flattened arrangement
handles every broadcast case without checking against PyTorch.

## Row-Wise Reduction

Validated self-tests: 2D last-dimension softmax and fixed-eps RMSNorm.

```python
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.tile((1, BLOCK_SIZE)), output.tile((1, BLOCK_SIZE))


tensors = (Tensor(2, other=float("-inf")), Tensor(2))
```

Use `other=float("-inf")` for max-based reductions such as softmax padding.
Cast to fp32 before reductions when precision matters.

For non-contiguous 2D reduction inputs, this package validates an explicit
`.contiguous()` fallback, not native arbitrary-stride reduction.

## Spatial / Pooling Pattern

Use `ravel()`, `flatten()`, and `tile()` to map spatial windows onto 1D blocks.
Check an existing pooling/convolution example before writing new arrangements.
