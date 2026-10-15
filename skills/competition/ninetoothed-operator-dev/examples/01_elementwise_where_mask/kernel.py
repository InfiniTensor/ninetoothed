"""
Masked elementwise add.

operation : out[i] = a[i] + b[i]  where  mask[i] else  a[i]
shapes    : a (M, N), b (M, N), mask (M, N) bool

Design notes
------------
* Use tile((1, BLOCK_SIZE)) for row-wise processing over a 2-D input.
* Mask fill value 0.0 declared on the mask Tensor via `other=`; masked-off
  mask lanes are treated as False (0), so ntl.where picks `a` correctly.
* fp16 / bf16: pure elementwise add has no catastrophic cancellation, so no
  upcast needed.  The output dtype follows the accumulation dtype (a.dtype).
* Broadcast from (1, N) to (M, N) is handled in the wrapper via
  mask.expand_as(a) before calling the kernel — wrapper fast-path, because
  the in-arrangement broadcast path would require different tile ranks for
  mask vs a/b, complicating the arrangement.
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(a, b, mask, out, BLOCK_SIZE=BLOCK_SIZE):
    """Tile all four tensors identically: one row per program."""
    return (
        a.tile((1, BLOCK_SIZE)),
        b.tile((1, BLOCK_SIZE)),
        mask.tile((1, BLOCK_SIZE)),
        out.tile((1, BLOCK_SIZE)),
    )


def application(a, b, mask, out):
    out = ntl.where(mask, a + b, a)  # noqa: F841


# Tensor(2) = 2-D symbolic tensor with default fill.
# mask Tensor: bool; OOB lanes default to False via other=0 (no effect on
# valid elements since mask is bool and ntl.where treats 0 as False).
_TENSORS = (Tensor(2), Tensor(2), Tensor(2, other=0), Tensor(2))

kernel = ninetoothed.make(arrangement, application, _TENSORS)
