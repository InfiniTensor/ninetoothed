import itertools
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import reduction as R  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"
SHAPES = [(8, 1024), (8, 4096), (16, 777)]  # 777 Non-power-of-two.
DTYPES = [torch.float16, torch.float32]
MODES = ["none", "sum", "mean"]
# Row sums over N fp16 values accumulate error; keep fp16 generous, fp32 tight.
TOL = {
    torch.float16: dict(atol=3e-1, rtol=1e-2),
    torch.float32: dict(atol=1e-4, rtol=1e-4),
}


@pytest.mark.parametrize(
    "shape,dtype,mode", list(itertools.product(SHAPES, DTYPES, MODES))
)
def test_correctness(shape, dtype, mode):
    x = torch.randn(*shape, dtype=dtype, device=DEVICE)
    got = R.row_reduce(x, reduction=mode)
    expected = R.reference(x, reduction=mode)
    tol = (
        dict(atol=1e-2, rtol=1e-2)
        if mode == "mean" and dtype == torch.float16
        else TOL[dtype]
    )
    torch.testing.assert_close(got, expected, **tol)
