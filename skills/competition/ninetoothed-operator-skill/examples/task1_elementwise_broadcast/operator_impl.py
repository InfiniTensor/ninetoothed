"""
T1: Masked Add with Broadcast — NineToothed Implementation.

Semantics: C[i,j] = mask[i,j] ? (A[i,j] + B[j]) : A[i,j]
where B (shape (N,)) broadcasts along the row dimension of A (shape (M,N)).

Fix: Expand B to (M, N) in PyTorch before passing to the kernel.
This avoids mixed-dimensionality tiling issues in the arrangement.
All four tensors are now 2D and tiled identically.

Pattern: Elementwise with Broadcast (Pattern B)
Reference: ninetoothed-examples elementwise kernels
"""

import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK_M = block_size()
BLOCK_N = block_size()


def masked_add_broadcast_arrangement(A, B2d, mask, C):
    """Arrange all 2D tensors with identical (BLOCK_M, BLOCK_N) tiling.

    A:    (M, N) -> tile (BLOCK_M, BLOCK_N)
    B2d:  (M, N) -> tile (BLOCK_M, BLOCK_N)  [pre-expanded from (N,)]
    mask: (M, N) -> tile (BLOCK_M, BLOCK_N)
    C:    (M, N) -> tile (BLOCK_M, BLOCK_N)
    """
    C_tiled = C.tile((BLOCK_M, BLOCK_N))
    A_tiled = A.tile((BLOCK_M, BLOCK_N))
    B2d_tiled = B2d.tile((BLOCK_M, BLOCK_N))
    mask_tiled = mask.tile((BLOCK_M, BLOCK_N))
    return A_tiled, B2d_tiled, mask_tiled, C_tiled


def masked_add_broadcast_application(A, B2d, mask, C):
    """Per-tile computation: C = A + B2d * mask (mask acts as 0/1)."""
    C = A + B2d * mask  # noqa: F841


TENSORS = (Tensor(2), Tensor(2), Tensor(2), Tensor(2))

_masked_add_broadcast_kernel = ninetoothed.make(
    masked_add_broadcast_arrangement,
    masked_add_broadcast_application,
    TENSORS,
)


def masked_add_broadcast(A, B, mask, C):
    """Public API: compute C = mask ? (A + B) : A.

    Expands B from (N,) to (M, N) before kernel launch to avoid
    mixed-dimensionality tiling in the NineToothed arrangement.

    Args:
        A: torch.Tensor, shape (M, N), float32, on CUDA.
        B: torch.Tensor, shape (N,), float32, on CUDA (broadcast vector).
        mask: torch.Tensor, shape (M, N), bool, on CUDA.
        C: torch.Tensor, shape (M, N), float32, on CUDA (output, pre-allocated).

    Returns:
        None (result written to C in-place).
    """
    # Expand B from (N,) to (M, N) for identical-dimension kernel launch
    B_expanded = B.unsqueeze(0).expand(A.shape[0], -1)
    _masked_add_broadcast_kernel(A, B_expanded, mask, C)
