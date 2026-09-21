import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(x, scale, out):
    return x.tile((1, 32)), scale, out.tile((1, 32))


def application(x, scale, out):
    mean = ntl.sum(x, axis=1) / scale
    out = x - mean[:, None]  # noqa: F841


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("constexpr", [False, True])
def test_scalar_in_row_expression_is_not_loaded_as_pointer(constexpr):
    scalar = Tensor(
        0, dtype="float32", constexpr=constexpr, value=32.0 if constexpr else None
    )
    kernel = ninetoothed.make(
        arrangement, application, (Tensor(2), scalar, Tensor(2)), max_num_configs=1
    )
    x = torch.randn((3, 32), device="cuda")
    out = torch.full_like(x, -12345)
    kernel(x, 32.0, out)
    torch.testing.assert_close(out, x - x.mean(1, keepdim=True))
