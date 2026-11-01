"""
pixel_unshuffle wrapper (space-to-depth).

Strategy: wrapper fast-path (see kernel.py design notes).
  1. Make x contiguous (handles non-contiguous inputs).
  2. Reindex via view + permute — O(1) metadata ops.
  3. Call a flat elementwise copy kernel to materialise the output
     as a fresh contiguous NCHW tensor.

This gives correct semantics for ALL inputs (contiguous and non-contiguous)
while keeping the arrangement trivially verifiable.
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

import torch

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import kernel as _kernel_module

_kernel = _kernel_module.kernel


def pixel_unshuffle(x: torch.Tensor, downscale_factor: int = 2) -> torch.Tensor:
    """Space-to-depth: (B, C, H, W) → (B, C·r², H/r, W/r).

    Equivalent to torch.nn.functional.pixel_unshuffle(x, downscale_factor).

    Args:
        x               : 4-D NCHW float tensor; H and W must be divisible by r.
        downscale_factor: r ≥ 1.

    Returns:
        Contiguous NCHW tensor of shape (B, C*r*r, H//r, W//r).
    """
    r = downscale_factor
    B, C, H, W = x.shape

    if H % r != 0 or W % r != 0:
        raise ValueError(
            f"H={H} and W={W} must both be divisible by downscale_factor={r}."
        )

    # Ensure contiguous (handles non-contiguous / transposed inputs).
    x = x.contiguous()

    # Re-index: (B, C, H, W) → (B, C, H/r, r, W/r, r) → permute → (B, C*r*r, H/r, W/r).
    reindexed = (
        x.view(B, C, H // r, r, W // r, r)
        .permute(0, 1, 3, 5, 2, 4)  # (B, C, r, r, H/r, W/r).
        .reshape(B, C * r * r, H // r, W // r)
    )
    # Reindexed is contiguous after reshape; materialise via kernel copy.
    src_flat = reindexed.flatten()
    dst_flat = torch.empty_like(src_flat)
    block = min(max(1, _next_pow2(src_flat.numel())), 4096)
    # For very large tensors, iterate blocks (BLOCK_SIZE caps at 4096).
    _kernel(src_flat, dst_flat, BLOCK_SIZE=block)

    return dst_flat.view(B, C * r * r, H // r, W // r)


def _next_pow2(x: int) -> int:
    p = 1

    while p < x:
        p <<= 1
    return p
