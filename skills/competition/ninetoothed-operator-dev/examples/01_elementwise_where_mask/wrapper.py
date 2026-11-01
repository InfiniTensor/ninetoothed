"""
PyTorch-facing wrapper for the masked elementwise add kernel.

Accepts mask of shape (M, N) or a broadcastable shape (e.g. (1, N));
broadcasts mask to (M, N) before the kernel call (wrapper fast-path).
"""

import pathlib as _pathlib
import sys as _sys

import torch

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import kernel as _kernel_module

_kernel = _kernel_module.kernel


def masked_add(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute a + b where mask is True, else a.

    Args:
        a    : (..., M, N) contiguous float tensor.
        b    : same shape as a.
        mask : bool tensor broadcastable to a.shape.

    Returns:
        out  : same shape and dtype as a.
    """
    # Ensure contiguous; expand mask to full shape (wrapper fast-path).
    a = a.contiguous()
    b = b.contiguous()
    mask = mask.expand_as(a).contiguous()

    out = torch.empty_like(a)
    # BLOCK_SIZE must be >= number of columns so one tile covers a whole row;
    # for large N choose the next power-of-two up to a hardware limit.
    N = a.shape[-1]
    block = min(max(1, _next_pow2(N)), 4096)
    _kernel(a, b, mask, out, BLOCK_SIZE=block)

    return out


def _next_pow2(x: int) -> int:
    p = 1

    while p < x:
        p <<= 1
    return p
