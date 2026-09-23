# Elementwise / Broadcast family

Output element depends only on same-index inputs. The arrangement just tiles
every tensor the same way; the application writes one expression.

> Provenance tags — calibrate trust per claim: **[E]** exercised end-to-end on
> GPU in this project's episode runs (raw run logs kept with the companion
> harness, not committed to this package — re-run to reproduce) · **[S]** verified
> against the `ninetoothed==0.25.0` source, not executed here · **[I]**
> inferred from adjacent patterns — re-verify before relying on it.
>
> A family-level **[E]** certifies only that the family was solved end-to-end —
> NOT that a specific `ntl.<fn>` call or a specific tiling recipe below was the
> one the winning solution used (that was not captured). Those carry their own
> [S]/[I] tag.

## 1D flat (add, mul, relu, gelu) — [E, family-level]

**[E: in the 2026-07-11 repo-aware A/B run, skill-equipped agents completed the
four train elementwise operators (add, mul_broadcast, relu, gelu), checked
against the harness's PyTorch oracle on GPU — the family works end-to-end. Raw
run logs are kept with the companion harness (not committed here). Which `ntl`
calls / tiling the solutions used internally was not captured; see the per-item
tags below.]**

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

def arrangement(input, other, output, BLOCK_SIZE=BLOCK_SIZE):
    return (input.tile((BLOCK_SIZE,)),
            other.tile((BLOCK_SIZE,)),
            output.tile((BLOCK_SIZE,)))

def application(input, other, output):
    output = input + other  # noqa: F841

kernel = ninetoothed.make(arrangement, application, tuple(Tensor(1) for _ in range(3)))
```

Torch wrapper (the standard invocation pattern):

```python
import torch
def add(input, other):
    output = torch.empty_like(input)
    kernel(input, other, output, BLOCK_SIZE=1024)   # constexpr passed at call
    return output
```

For an N-D input that is contiguous, flatten in the wrapper and reshape back
(this is how `silu`/`swiglu` work in the repo):

```python
def silu(x):
    x_flat = x.flatten()
    out_flat = torch.empty_like(x_flat)
    kernel(x_flat, out_flat, BLOCK_SIZE=1024)
    return out_flat.view_as(x)
```

## Unary via libdevice / ntl math

`application` can call `ntl.<fn>`. Common unary/binary math: `ntl.exp`,
`ntl.where`, `ntl.maximum`, `ntl.sigmoid`, `ntl.tanh` **[I — the family was
solved on GPU (above), but which of these the solutions actually emitted was
not captured, and each `ntl.<fn>` resolves dynamically at trace time, so
per-function availability is version-dependent; re-verify on your version]**.
If `ntl.tanh` fails to translate, hand-roll it from `ntl.exp`
(`(e^{2x}-1)/(e^{2x}+1)`). Example ReLU and GELU-ish:

```python
def application(input, output):
    output = ntl.where(input > 0, input, 0.0)  # noqa: F841
```

## Broadcast (A[M,1] op B[1,N]) — [I]

`mul_broadcast` (a train task) was solved on GPU in the 2026-07-11 run, but
whether the solution used this in-arrangement 2-D recipe or a wrapper-side
`expand()`+`contiguous()` was not captured — treat the recipe below as
inferred, and confirm with `simulate_arrangement` before relying on it.

Tile both operands and the output on the 2-D grid; rely on Triton scalar/row
broadcast inside the tile. Keep the broadcasted axis size 1 in the source
`Tensor` and let the expression broadcast:

```python
def arrangement(a, b, output, BM=Symbol("BM", constexpr=True),
                BN=Symbol("BN", constexpr=True)):
    return (a.tile((BM, 1)), b.tile((1, BN)), output.tile((BM, BN)))
```

If the broadcast shapes do not divide evenly, pass `other=` so out-of-range
loads are well-defined (see mask below).

## Mask / padding (`other=`) — [S]

When a tile can read past the tensor edge (non-divisible shape, or a masked op),
declare the fill value on the **source** `Tensor`:

```python
tensors = (Tensor(2, other=0.0), Tensor(2))   # OOB loads read 0.0
```

Use `float("-inf")` for max/softmax inputs, `0.0` for sum/add. To apply an
explicit boolean mask inside the application use `ntl.where(cond, x, fill)`.

## fp16 / bf16 numeric care — [S]

Pure elementwise add/mul in fp16 is fine. Only upcast when the op is
numerically sensitive (involves exp, division by small numbers, or long sums) —
see `reduction.md`. Do not blanket-upcast; it wastes registers.

## Pitfalls (see common-errors.md for fixes)

- Forgetting `# noqa: F841` on the `output = ...` line.
- Using `tile((BLOCK_SIZE,))` on a 2-D tensor when you meant row-wise — that
  tiles dim 0. For per-row elementwise use `tile((1, BLOCK_SIZE))`.
- Passing a multi-dim tensor to a `Tensor(1)` kernel without flattening.
