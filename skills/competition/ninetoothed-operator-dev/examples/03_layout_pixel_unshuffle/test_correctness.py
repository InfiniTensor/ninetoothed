"""
Correctness tests for pixel_unshuffle.

Key coverage:
* Multiple downscale factors (2, 3, 4).
* Contiguous AND non-contiguous (transposed) inputs — the defining feature of
  the layout-sensitive family.
* fp16 and fp32.
* Non-power-of-two spatial dimensions.
"""

import itertools
import pathlib as _pathlib
import sys as _sys

import pytest
import torch
import torch.nn.functional as F

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
from wrapper import pixel_unshuffle

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"
# (B, C, H, W) — H and W must be divisible by the downscale factor used.
# Using multiples of 12 to cover factors 2, 3, 4 with a single shape set.
SHAPES = [(1, 2, 12, 12), (2, 4, 24, 24), (1, 1, 12, 36)]
FACTORS = [2, 3, 4]
DTYPES = [torch.float32, torch.float16]
TOL = {
    torch.float32: dict(atol=0, rtol=0),  # Pure copy — must be exact.
    torch.float16: dict(atol=0, rtol=0),
}


@pytest.mark.parametrize(
    "shape,factor,dtype,contig",
    list(itertools.product(SHAPES, FACTORS, DTYPES, [True, False])),
)
def test_pixel_unshuffle(shape, factor, dtype, contig):
    B, C, H, W = shape

    if H % factor != 0 or W % factor != 0:
        pytest.skip(f"shape {shape} not divisible by factor {factor}")

    if not contig:
        # Allocate as (B, C, W, H) and transpose the last two dims -> logical shape
        # (B, C, H, W) with non-contiguous storage. A SINGLE transpose is used
        # deliberately (transposing twice would return a contiguous view).
        x = torch.randn(B, C, W, H, dtype=dtype, device=DEVICE).transpose(-1, -2)
        assert not x.is_contiguous(), "Expected non-contiguous tensor for this branch."
        assert x.shape == (B, C, H, W)
    else:
        x = torch.randn(B, C, H, W, dtype=dtype, device=DEVICE)

    expected = F.pixel_unshuffle(x.contiguous(), factor)  # Torch reference.
    got = pixel_unshuffle(x, factor)

    torch.testing.assert_close(got, expected, **TOL[dtype])
    assert got.is_contiguous(), "Output must be contiguous."
    assert got.shape == (B, C * factor * factor, H // factor, W // factor)


def test_not_supported_non_divisible():
    """H not divisible by factor → raises ValueError (documented limitation)."""
    x = torch.randn(1, 1, 5, 4, device=DEVICE)

    with pytest.raises(ValueError, match="divisible"):
        pixel_unshuffle(x, downscale_factor=2)
