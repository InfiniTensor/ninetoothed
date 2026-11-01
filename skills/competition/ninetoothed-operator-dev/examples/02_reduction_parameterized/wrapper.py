"""
PyTorch-facing wrapper for the reduction kernel.

Implements reduction='none' | 'sum' | 'mean' over the last dim of x.

'none'  → returns x as-is (no kernel; included to match torch API shape).
'sum'   → calls the row-sum kernel once.
'mean'  → calls the row-sum kernel, then divides by x.shape[-1].
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

import torch

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import kernel as _kernel_module

_kernel = _kernel_module.kernel


def reduce_last_dim(
    x: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Reduce x along the last dimension.

    Args:
        x         : (M, N) or (B, M, N) float tensor, contiguous or not.
        reduction : 'none' | 'sum' | 'mean'

    Returns:
        'none' → same shape as x.
        'sum' / 'mean' → x with last dim removed: (M,) or (B, M).
    """
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"Unsupported reduction={reduction!r}.")

    x = x.contiguous()
    orig_shape = x.shape
    N = orig_shape[-1]

    if reduction == "none":
        return x.clone()

    # Flatten all leading dims into a single M axis.
    M = x.numel() // N
    x_2d = x.view(M, N)

    out = torch.empty(M, dtype=x.dtype, device=x.device)
    # BLOCK_SIZE covers the entire row in one tile (up to 4096).
    block = min(max(1, _next_pow2(N)), 4096)
    _kernel(x_2d, out, BLOCK_SIZE=block)

    if reduction == "mean":
        out = out / N

    # Restore leading dims (drop the last one).
    return out.view(orig_shape[:-1])


def _next_pow2(x: int) -> int:
    p = 1

    while p < x:
        p <<= 1
    return p
