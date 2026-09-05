import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "rounding_mode",
    [
        None,
        pytest.param(
            "trunc", marks=pytest.mark.skip(reason="TODO: Test for `trunc` mode later.")
        ),
        "floor",
    ],
)
@pytest.mark.parametrize(*generate_arguments())
def test_div(shape, rounding_mode, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.div(input, other, rounding_mode=rounding_mode)
    reference_output = torch.div(input, other, rounding_mode=rounding_mode)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
