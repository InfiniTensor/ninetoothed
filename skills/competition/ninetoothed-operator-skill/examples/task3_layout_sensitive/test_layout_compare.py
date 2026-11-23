"""
T3 Cross-Layout Consistency Test.

Verifies that the same logical data in different memory layouts
produces identical GELU output (within float32 precision).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from operator_impl import gelu


def test_contiguous_vs_non_contiguous_identical():
    """Same data, contiguous vs transposed layout — must match."""
    device = "cuda"
    M, N = 512, 512

    data = torch.randn((N, M), dtype=torch.float32, device=device)

    # Contiguous version
    X_contig = data.T.contiguous()
    out_contig = torch.empty_like(X_contig)
    gelu(X_contig, out_contig)

    # Non-contiguous version
    X_noncontig = data.T
    out_noncontig = torch.empty_like(X_noncontig)
    gelu(X_noncontig, out_noncontig)

    # Compare
    diff = (out_contig - out_noncontig.contiguous()).abs().max().item()
    print(f"Max difference contiguous vs non-contiguous: {diff}")

    torch.testing.assert_close(out_contig, out_noncontig, atol=1e-5, rtol=1e-3)
    print("✓ Contiguous and non-contiguous produce identical results")


def test_sliced_vs_contiguous_subset():
    """Sliced view of contiguous must equal sliced output of contiguous."""
    device = "cuda"
    M, N = 512, 512

    data = torch.randn((M, N), dtype=torch.float32, device=device)

    # Full contiguous
    out_full = torch.empty_like(data)
    gelu(data, out_full)

    # Sliced view
    X_sliced = data[::2, :]
    out_sliced = torch.empty_like(X_sliced)
    gelu(X_sliced, out_sliced)

    # Verify: contiguous[::2] == sliced_output
    torch.testing.assert_close(out_full[::2, :], out_sliced, atol=1e-5, rtol=1e-3)
    print("✓ Sliced output matches contiguous subset")


if __name__ == "__main__":
    test_contiguous_vs_non_contiguous_identical()
    test_sliced_vs_contiguous_subset()
    print("✓ All cross-layout consistency tests passed")
