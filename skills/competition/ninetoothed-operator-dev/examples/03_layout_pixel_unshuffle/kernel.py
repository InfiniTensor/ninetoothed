"""
Space-to-depth (pixel_unshuffle) as a layout-sensitive operator.

operation : out[b, c*r*r + (kh*r + kw), h, w]
              = x[b, c, h*r + kh, w*r + kw]
            with downscale_factor r, input (B,C,H,W) → output (B,C*r*r,H/r,W/r)

Design decision
---------------
This operation is a pure index re-mapping with NO arithmetic — there is nothing
to "compute" in the application beyond a load + store.

In NineToothed this is naturally expressed with the WRAPPER FAST-PATH:
  x.view(B, C, H//r, r, W//r, r).permute(0,1,3,5,2,4).reshape(B, C*r*r, H//r, W//r)

This is preferred over a complex in-arrangement ravel/flatten chain because:
1. `view` is O(1) (changes metadata, not data) when the input is contiguous.
2. The resulting tensor IS contiguous → a trivial elementwise copy kernel works.
3. Attempting the full re-indexing in the arrangement would require a nested
   tile + permute + ravel chain that is harder to verify and offers no
   performance advantage (the dominant cost is the memory copy either way).

The kernel below is the ELEMENTWISE COPY that moves data after the view/permute.
It exists to demonstrate the NineToothed invocation pattern for layout ops
and to produce a runnable, testable artifact.

For non-contiguous inputs, the wrapper calls .contiguous() first so the
view/permute sequence is always valid.
"""

import ninetoothed
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(src, dst, BLOCK_SIZE=BLOCK_SIZE):
    """Flat 1-D copy; both tensors are 1-D after flatten in the wrapper."""
    return src.tile((BLOCK_SIZE,)), dst.tile((BLOCK_SIZE,))


def application(src, dst):
    dst = src  # noqa: F841


_TENSORS = (Tensor(1), Tensor(1))

kernel = ninetoothed.make(arrangement, application, _TENSORS)
