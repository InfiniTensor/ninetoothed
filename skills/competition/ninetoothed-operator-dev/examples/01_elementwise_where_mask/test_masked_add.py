import itertools
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import masked_add as M  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"
# (B, N): include a non-power-of-two N.
SHAPES = [(8, 1024), (8, 4095), (3, 777)]
DTYPES = [torch.float16, torch.float32]
TOL = {
    torch.float16: dict(atol=1e-3, rtol=1e-3),
    torch.float32: dict(atol=1e-5, rtol=1e-5),
}


@pytest.mark.parametrize("shape,dtype", list(itertools.product(SHAPES, DTYPES)))
def test_correctness(shape, dtype):
    b_rows, n = shape
    a = torch.randn(b_rows, n, dtype=dtype, device=DEVICE)
    b = torch.randn(1, n, dtype=dtype, device=DEVICE)  # Broadcast over rows.
    mask = (torch.rand(b_rows, n, device=DEVICE) > 0.5).to(dtype)
    got = M.masked_add(a, b, mask)
    expected = M.reference(a, b, mask)
    torch.testing.assert_close(got, expected, **TOL[dtype])


def test_no_negative_transfer_on_full_mask():
    a = torch.randn(4, 256, device=DEVICE)
    b = torch.randn(1, 256, device=DEVICE)
    mask = torch.ones(4, 256, device=DEVICE)
    torch.testing.assert_close(M.masked_add(a, b, mask), a + b, atol=1e-5, rtol=1e-5)
