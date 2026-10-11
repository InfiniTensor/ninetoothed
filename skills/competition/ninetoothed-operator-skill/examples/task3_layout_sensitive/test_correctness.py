"""
T3 Correctness Test: Layout-Sensitive GELU.

Tests 4 layout variants:
  1. contiguous           — standard row-major
  2. transposed (.T)      — column-major strides
  3. sliced ([::2, :])    — non-unit stride on outer dim
  4. offset ([:, 1:])     — non-zero base pointer offset

All layout variants must produce identical logical results.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from operator_impl import gelu


def pytorch_reference(X):
    """PyTorch GELU with tanh approximation."""
    return nn.GELU(approximate="tanh")(X)


# ---- Contiguous baseline ----


@pytest.mark.parametrize(
    "M,N",
    [
        (256, 256),
        (1024, 1024),
        (2, 2),
        (257, 257),
    ],
)
def test_gelu_contiguous(M, N):
    """Baseline: contiguous input."""
    device = "cuda"
    X = torch.randn((M, N), dtype=torch.float32, device=device)

    expected = pytorch_reference(X)
    output = torch.empty_like(X)
    gelu(X, output)

    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-3)


# ---- Transposed (non-contiguous) ----


@pytest.mark.parametrize(
    "M,N",
    [
        (256, 256),
        (1024, 512),
        (257, 257),
    ],
)
def test_gelu_transposed(M, N):
    """Non-contiguous: transposed input (column-major strides)."""
    device = "cuda"
    X_base = torch.randn((N, M), dtype=torch.float32, device=device)
    X = X_base.T  # shape (M, N), strides (1, M) — non-contiguous

    expected = pytorch_reference(X)
    output = torch.empty_like(X)
    gelu(X, output)

    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-3)


# ---- Sliced (non-unit stride) ----


@pytest.mark.parametrize(
    "M,N",
    [
        (256, 128),
        (1024, 512),
    ],
)
def test_gelu_sliced(M, N):
    """Non-contiguous: sliced input (stride[0] = 2× original)."""
    device = "cuda"
    X_full = torch.randn((M * 2, N), dtype=torch.float32, device=device)
    X = X_full[::2, :]  # shape (M, N), doubled stride on dim 0

    expected = pytorch_reference(X)
    output = torch.empty_like(X)
    gelu(X, output)

    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-3)


# ---- Offset (non-zero base pointer) ----


@pytest.mark.parametrize(
    "M,N",
    [
        (256, 128),
        (1024, 512),
    ],
)
def test_gelu_offset(M, N):
    """Non-contiguous: offset input (non-zero storage_offset)."""
    device = "cuda"
    X_full = torch.randn((M, N + 16), dtype=torch.float32, device=device)
    X = X_full[:, 1 : N + 1]  # shape (M, N), offset from base pointer

    expected = pytorch_reference(X)
    output = torch.empty_like(X)
    gelu(X, output)

    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-3)


# ---- Cross-layout consistency ----


def test_gelu_cross_layout_consistency():
    """Same logical data in 4 layouts must produce identical results."""
    device = "cuda"
    M, N = 256, 256
    data = torch.randn((M, N), dtype=torch.float32, device=device)

    # Layout 1: contiguous
    X1 = data
    out1 = torch.empty_like(X1)
    gelu(X1, out1)

    # Layout 2: transposed (force non-contiguous via round-trip)
    X2 = data.T.contiguous().T
    out2 = torch.empty_like(X2)
    gelu(X2, out2)

    # Layout 3: sliced
    X3 = data[::2, :]
    out3 = torch.empty_like(X3)
    gelu(X3, out3)

    # Verify contiguous[::2] == sliced_output
    torch.testing.assert_close(out1[::2, :], out3, atol=1e-5, rtol=1e-3)

    # Verify contiguous == transposed (after re-transposing back)
    torch.testing.assert_close(
        out1, out2.contiguous().T.contiguous().T, atol=1e-5, rtol=1e-3
    )
