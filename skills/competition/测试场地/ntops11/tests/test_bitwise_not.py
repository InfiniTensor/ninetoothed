import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments(False))
def test_bitwise_not(shape, dtype, device, rtol, atol):
    if dtype == torch.bool:
        prob = 0.5
        input = torch.rand(shape, dtype=torch.float32, device=device) > prob
    else:
        upper_bound = 10
        input = torch.randint(
            -upper_bound, upper_bound, size=shape, dtype=dtype, device=device
        )

    ninetoothed_output = ntops.torch.bitwise_not(input)
    reference_output = torch.bitwise_not(input)

    assert torch.equal(ninetoothed_output, reference_output)
