"""
T1 Correctness Test: Masked Add with Broadcast.

Tests:
  - Normal shapes
  - Boundary: M=1, N=1
  - Non-aligned BLOCK_SIZE
  - Mask patterns: all_true, all_false, checkerboard
  - Non-contiguous input (transposed) combined with broadcast
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from operator_impl import masked_add_broadcast


def pytorch_reference(A, B, mask):
    """PyTorch reference: where(mask, A+B, A)."""
    return torch.where(mask, A + B, A)


# ---- Parameterized shape tests ----


@pytest.mark.parametrize(
    "M,N",
    [
        (1024, 1024),
        (1, 1024),
        (1024, 1),
        (1, 1),
        (2048, 2048),
        (256, 256),
        (257, 257),
    ],
)
def test_masked_add_broadcast(M, N):
    """Correctness across shape spectrum including boundary conditions."""
    device = "cuda"
    A = torch.randn((M, N), dtype=torch.float32, device=device)
    B = torch.randn((N,), dtype=torch.float32, device=device)
    mask = torch.rand((M, N), device=device) > 0.5

    expected = pytorch_reference(A, B, mask)
    output = torch.empty_like(A)
    masked_add_broadcast(A, B, mask, output)

    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-3)


# ---- Mask pattern tests ----


@pytest.mark.parametrize(
    "mask_pattern",
    [
        "all_true",
        "all_false",
        "checkerboard",
    ],
)
def test_masked_add_broadcast_mask_patterns(mask_pattern):
    """Verify degenerate mask cases behave correctly."""
    device = "cuda"
    M, N = 512, 512
    A = torch.randn((M, N), dtype=torch.float32, device=device)
    B = torch.randn((N,), dtype=torch.float32, device=device)

    if mask_pattern == "all_true":
        mask = torch.ones((M, N), dtype=torch.bool, device=device)
    elif mask_pattern == "all_false":
        mask = torch.zeros((M, N), dtype=torch.bool, device=device)
    else:
        mask = torch.zeros((M, N), dtype=torch.bool, device=device)
        mask[::2, ::2] = True
        mask[1::2, 1::2] = True

    expected = pytorch_reference(A, B, mask)
    output = torch.empty_like(A)
    masked_add_broadcast(A, B, mask, output)

    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-3)


# ---- Non-contiguous input test ----


def test_masked_add_broadcast_non_contiguous():
    """Transposed A — tests stride handling with broadcast."""
    device = "cuda"
    N, M = 512, 256
    A_base = torch.randn((M, N), dtype=torch.float32, device=device)
    A = A_base.T  # non-contiguous view
    B = torch.randn((M,), dtype=torch.float32, device=device)
    mask_base = torch.rand((M, N), device=device) > 0.5
    mask = mask_base.T

    expected = pytorch_reference(A, B, mask)
    output = torch.empty_like(A)
    masked_add_broadcast(A, B, mask, output)

    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-3)


# ---- dtype variant test ----


def test_masked_add_broadcast_float16():
    """float16 variant — relaxed tolerance."""
    device = "cuda"
    M, N = 256, 256
    A = torch.randn((M, N), dtype=torch.float16, device=device)
    B = torch.randn((N,), dtype=torch.float16, device=device)
    mask = torch.rand((M, N), device=device) > 0.5

    expected = pytorch_reference(A.float(), B.float(), mask).half()
    output = torch.empty((M, N), dtype=torch.float16, device=device)
    masked_add_broadcast(A, B, mask, output)

    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-2)
