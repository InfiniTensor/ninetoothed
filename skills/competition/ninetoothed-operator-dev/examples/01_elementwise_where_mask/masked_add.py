"""Example 1 — masked elementwise add with broadcast.

Family: elementwise / broadcast.
Confidence: HIGH — kernel follows the `add` pattern from the NineToothed repo's
own `tests/test_add.py`, plus an ntl.where mask; broadcast is resolved in the
wrapper.

out = where(mask, a + broadcast(b), 0)
"""

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(a, b, mask, output, BLOCK_SIZE=BLOCK_SIZE):
    return (
        a.tile((BLOCK_SIZE,)),
        b.tile((BLOCK_SIZE,)),
        mask.tile((BLOCK_SIZE,)),
        output.tile((BLOCK_SIZE,)),
    )


def application(a, b, mask, output):
    output = ntl.where(mask != 0, a + b, 0.0)  # noqa: F841


_kernel = ninetoothed.make(arrangement, application, tuple(Tensor(1) for _ in range(4)))


def masked_add(a, b, mask, block_size=1024):
    """Compute the masked add — a: (..., N); b: broadcastable to a; mask: broadcastable to a (0/1).

    Broadcast is materialized in the wrapper (a view + contiguous), then the
    kernel runs on flat 1-D tensors — NineToothed auto-masks the tail block,
    so N need not be a multiple of block_size.
    """
    b_exp = b.expand_as(a).contiguous()
    mask_f = mask.to(a.dtype).expand_as(a).contiguous()

    a_flat = a.contiguous().flatten()
    b_flat = b_exp.flatten()
    m_flat = mask_f.flatten()
    out_flat = torch.empty_like(a_flat)

    _kernel(a_flat, b_flat, m_flat, out_flat, BLOCK_SIZE=block_size)

    return out_flat.view_as(a)


def reference(a, b, mask):
    return torch.where(
        mask.bool().expand_as(a), a + b.expand_as(a), torch.zeros_like(a)
    )
