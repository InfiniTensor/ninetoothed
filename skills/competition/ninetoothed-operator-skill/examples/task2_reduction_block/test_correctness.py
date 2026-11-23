"""
T2 Correctness Test: Tiled Softmax with Numerical Stability.

Tests:
  - Normal shapes
  - Very long sequence (N=32768) — tiling stress
  - Single-element reduction (N=1)
  - Non-aligned BLOCK_SIZE (N=257)
  - Numerical stability: large positive/negative/mixed values
  - Row sums equal 1.0
  - All-equal input → uniform output (1/N)
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from operator_impl import tiled_softmax


def pytorch_reference(X):
    """PyTorch reference — numerically stable softmax along dim=-1."""
    return F.softmax(X, dim=-1)


# ---- Parameterized shape tests ----


@pytest.mark.parametrize(
    "M,N",
    [
        (256, 256),
        (1024, 1024),
        (1, 32768),
        (1024, 1),
        (32, 128),
        (1024, 257),
    ],
)
def test_softmax_correctness(M, N):
    """Correctness across diverse shape configurations."""
    device = "cuda"
    X = torch.randn((M, N), dtype=torch.float32, device=device)

    expected = pytorch_reference(X)
    output = torch.empty_like(X)
    tiled_softmax(X, output)

    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-3)


# ---- Numerical stability tests ----


@pytest.mark.parametrize(
    "value_range",
    [
        "normal",
        "large_pos",
        "large_neg",
        "mixed_large",
    ],
)
def test_softmax_numerical_stability(value_range):
    """Verify max-subtraction prevents NaN/inf for extreme values."""
    device = "cuda"
    M, N = 128, 256

    if value_range == "normal":
        X = torch.randn((M, N), dtype=torch.float32, device=device)
    elif value_range == "large_pos":
        X = torch.full((M, N), 1e4, dtype=torch.float32, device=device)
    elif value_range == "large_neg":
        X = torch.full((M, N), -1e4, dtype=torch.float32, device=device)
    else:
        X = torch.randn((M, N), dtype=torch.float32, device=device) * 1e4

    expected = pytorch_reference(X)
    output = torch.empty_like(X)
    tiled_softmax(X, output)

    # Verify no NaN or inf in output
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains inf"

    # Verify rows sum to 1
    row_sums = output.sum(dim=-1)
    torch.testing.assert_close(
        row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-4
    )

    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-3)


# ---- Row sum invariant ----


def test_softmax_row_sums_to_one():
    """Every row of softmax output must sum to exactly 1.0."""
    device = "cuda"
    M, N = 1024, 1024
    X = torch.randn((M, N), dtype=torch.float32, device=device)

    output = torch.empty_like(X)
    tiled_softmax(X, output)

    row_sums = output.sum(dim=-1)
    torch.testing.assert_close(
        row_sums,
        torch.ones(M, dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=1e-4,
    )


# ---- Uniform input test ----


def test_softmax_all_equal_input():
    """If all inputs are equal, all outputs should be 1/N."""
    device = "cuda"
    M, N = 64, 128
    X = torch.full((M, N), 3.14159, dtype=torch.float32, device=device)

    output = torch.empty_like(X)
    tiled_softmax(X, output)

    expected_val = 1.0 / N
    torch.testing.assert_close(
        output,
        torch.full_like(output, expected_val),
        atol=1e-5,
        rtol=1e-4,
    )


# ---- Non-contiguous input ----


def test_softmax_non_contiguous():
    """Softmax on transposed input — verifies stride handling."""
    device = "cuda"
    N, M = 256, 128
    X_base = torch.randn((M, N), dtype=torch.float32, device=device)
    X = X_base.T  # non-contiguous

    expected = pytorch_reference(X)
    output = torch.empty_like(X)
    tiled_softmax(X, output)

    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-3)
