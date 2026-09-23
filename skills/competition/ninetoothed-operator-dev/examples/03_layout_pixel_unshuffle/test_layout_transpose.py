import itertools
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import layout_transpose as L  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"
# Non-square + non-power-of-two to exercise tail masking on a strided read.
SHAPES = [(64, 128), (128, 64), (96, 80), (130, 77)]
DTYPES = [torch.float16, torch.float32]
TOL = {
    torch.float16: dict(atol=1e-3, rtol=1e-3),
    torch.float32: dict(atol=1e-5, rtol=1e-5),
}


@pytest.mark.parametrize("shape,dtype", list(itertools.product(SHAPES, DTYPES)))
def test_transpose_noncontiguous(shape, dtype):
    x = torch.randn(*shape, dtype=dtype, device=DEVICE)
    got = L.transpose_nt(x)
    expected = L.reference_transpose(x)
    torch.testing.assert_close(got, expected, **TOL[dtype])
