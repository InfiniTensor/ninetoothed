import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import softmax_variants as S  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"
SHAPES = [(64, 1024), (64, 2048), (128, 777)]  # 777 Is non-power-of-two.
DTYPES = [torch.float16, torch.float32]
TOL = {
    torch.float16: dict(atol=1e-3, rtol=1e-3),
    torch.float32: dict(atol=1e-5, rtol=1e-5),
}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("variant", ["fast", "slow"])
def test_correctness(shape, dtype, variant):
    x = torch.randn(*shape, dtype=dtype, device=DEVICE)
    fn = S.softmax_fast if variant == "fast" else S.softmax_slow
    torch.testing.assert_close(fn(x), S.reference(x), **TOL[dtype])
