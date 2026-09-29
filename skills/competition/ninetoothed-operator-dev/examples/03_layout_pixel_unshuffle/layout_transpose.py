"""Example 3 — layout-sensitive: materialize a non-contiguous input.

Family: layout-sensitive (non-contiguous / stride / offset).
Confidence: MEDIUM-HIGH. The copy kernel is trivial; what this validates is that
NineToothed reads a NON-CONTIGUOUS input correctly (it gets the source strides
at call time). Transpose is the cleanest non-contiguous case and is the
primitive that space-to-depth ops like pixel_unshuffle are built from.

`transpose_nt(x)` copies the non-contiguous view `x.t()` into a contiguous
tensor through a NineToothed kernel — i.e. NineToothed does the non-contiguous
read, not torch. Compare against `x.t().contiguous()`.

pixel_unshuffle extension: it is the same idea applied to a 6-D permuted view
(`x.view(B,C,H//r,r,W//r,r).permute(0,1,3,5,2,4)`); the reference is provided
below for when that arrangement is added.
"""

import torch
import torch.nn.functional as F

import ninetoothed
import ninetoothed.language as ntl  # noqa: F401  (kept for parity with other examples)
from ninetoothed import Symbol, Tensor

BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", constexpr=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", constexpr=True)


def arrangement(input, output, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N):
    return input.tile((BLOCK_SIZE_M, BLOCK_SIZE_N)), output.tile(
        (BLOCK_SIZE_M, BLOCK_SIZE_N)
    )


def application(input, output):
    output = input  # noqa: F841   (identity copy; NineToothed honors input strides)


_copy = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2)))


def transpose_nt(x, block_m=32, block_n=32):
    """Materialize x.t() (a non-contiguous view) via a NineToothed copy."""
    xt = x.t()  # Non-contiguous (N, M).
    out = torch.empty(xt.shape, dtype=x.dtype, device=x.device)  # Contiguous.
    _copy(xt, out, BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n)

    return out


def reference_transpose(x):
    return x.t().contiguous()


def reference_pixel_unshuffle(x, r):
    # Oracle for the documented extension.
    return F.pixel_unshuffle(x, downscale_factor=r)
