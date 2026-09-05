import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "approximate",
    (
        "none",
        pytest.param(
            "tanh", marks=pytest.mark.skip(reason="TODO: Test for `tanh` mode later.")
        ),
    ),
)
@pytest.mark.parametrize(*generate_arguments())
def test_gelu(shape, approximate, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.gelu(input, approximate=approximate)
    reference_output = F.gelu(input, approximate=approximate)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
