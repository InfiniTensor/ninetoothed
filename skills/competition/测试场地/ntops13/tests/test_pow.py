import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_pow(shape, dtype, device, rtol, atol):
    # TODO: Test for `float16` later.
    if dtype is torch.float16:
        return

    input = torch.randn(shape, dtype=dtype, device=device)
    exponent = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.pow(input, exponent)
    reference_output = torch.pow(input, exponent)

    assert torch.allclose(
        ninetoothed_output, reference_output, rtol=rtol, atol=atol, equal_nan=True
    )
