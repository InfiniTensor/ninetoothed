"""
Correctness tests for masked_add.

Matrix: shape × dtype × layout × broadcast.
"""

import itertools
import pathlib as _pathlib
import sys as _sys

import pytest
import torch

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
from wrapper import masked_add

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"
TOL = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.float16: dict(atol=1e-3, rtol=1e-3),
    torch.bfloat16: dict(atol=1e-2, rtol=1e-2),
}

# (M, N): include a non-power-of-two shape.
SHAPES = [(64, 128), (64, 127), (32, 512)]
DTYPES = [torch.float32, torch.float16]


def reference(a, b, mask):
    return torch.where(mask, a + b, a)


@pytest.mark.parametrize(
    "shape,dtype,contig",
    list(itertools.product(SHAPES, DTYPES, [True, False])),
)
def test_masked_add(shape, dtype, contig):
    M, N = shape
    a = torch.randn(M, N, dtype=dtype, device=DEVICE)
    b = torch.randn(M, N, dtype=dtype, device=DEVICE)
    mask = torch.randint(0, 2, (M, N), dtype=torch.bool, device=DEVICE)

    if not contig:
        # Make a non-contiguous view via transpose then back.
        a = a.t().contiguous().t()  # Still (M,N) but non-contiguous.
        b = b.t().contiguous().t()
        assert not a.is_contiguous()

    expected = reference(a, b, mask)
    got = masked_add(a, b, mask)
    torch.testing.assert_close(got, expected, **TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
def test_broadcast_mask(dtype):
    """Mask shape (1, N) → broadcast to (M, N) in the wrapper."""
    M, N = 32, 64
    a = torch.randn(M, N, dtype=dtype, device=DEVICE)
    b = torch.randn(M, N, dtype=dtype, device=DEVICE)
    mask = torch.randint(0, 2, (1, N), dtype=torch.bool, device=DEVICE)

    expected = reference(a, b, mask.expand(M, N))
    got = masked_add(a, b, mask)
    torch.testing.assert_close(got, expected, **TOL[dtype])


def test_all_true_mask():
    """When mask is all-True, output == a + b."""
    a = torch.randn(64, 128, dtype=torch.float32, device=DEVICE)
    b = torch.randn(64, 128, dtype=torch.float32, device=DEVICE)
    mask = torch.ones(64, 128, dtype=torch.bool, device=DEVICE)
    torch.testing.assert_close(masked_add(a, b, mask), a + b, atol=1e-5, rtol=1e-5)


def test_all_false_mask():
    """When mask is all-False, output == a."""
    a = torch.randn(64, 128, dtype=torch.float32, device=DEVICE)
    b = torch.randn(64, 128, dtype=torch.float32, device=DEVICE)
    mask = torch.zeros(64, 128, dtype=torch.bool, device=DEVICE)
    torch.testing.assert_close(masked_add(a, b, mask), a, atol=0, rtol=0)
