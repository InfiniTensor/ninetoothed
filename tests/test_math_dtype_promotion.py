import pytest
import torch

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import make
from tests.utils import get_available_devices


def _arrangement(x, out):
    return x.tile((128,)), out.tile((128,))


def _exp_application(x, out):
    out = ntl.exp(x)  # noqa: F841


def _exp2_application(x, out):
    out = ntl.exp2(x)  # noqa: F841


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)
@pytest.mark.parametrize(
    ("application", "reference"),
    ((_exp_application, torch.exp), (_exp2_application, torch.exp2)),
)
def test_exponential_promotes_low_precision_without_narrowing_float64(
    device, dtype, application, reference
):
    kernel = make(
        _arrangement,
        application,
        (Tensor(1), Tensor(1)),
        backend="triton",
        max_num_configs=1,
    )
    x = torch.linspace(-4, 4, 257, dtype=dtype, device=device)
    out = torch.empty_like(x)
    kernel(x, out)
    expected = reference(x.double() if dtype == torch.float64 else x.float())
    torch.testing.assert_close(out, expected.to(dtype))


def _promoted_constructor_application(x, out):
    first = ntl.zeros(x.shape, dtype=x.dtype)
    promoted = first + 0.5
    value = ntl.full(out.shape, 1.5, dtype=promoted.dtype)
    out = value  # noqa: F841


@pytest.mark.parametrize("device", get_available_devices())
def test_constructor_uses_computed_floating_dtype_from_integer_tensor(device):
    kernel = make(
        _arrangement,
        _promoted_constructor_application,
        (Tensor(1), Tensor(1)),
        backend="triton",
        max_num_configs=1,
    )
    x = torch.zeros(257, dtype=torch.int32, device=device)
    out = torch.empty(257, dtype=torch.float32, device=device)
    kernel(x, out)
    torch.testing.assert_close(out, torch.full_like(out, 1.5))
