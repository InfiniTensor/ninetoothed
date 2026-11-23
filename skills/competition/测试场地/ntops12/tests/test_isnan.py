import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_isnan(shape, dtype, device, rtol, atol):
    def generate_nan_tensor(shape, dtype, device):
        nan_prob = 0.4
        prob_tensor = torch.rand(shape, device=device)
        mask = prob_tensor < nan_prob

        x = torch.randn(shape, dtype=dtype, device=device)
        x[mask] = float("nan")

        return x

    input = generate_nan_tensor(shape, dtype, device)

    ninetoothed_output = ntops.torch.isnan(input)
    reference_output = torch.isnan(input)

    assert torch.equal(ninetoothed_output, reference_output)
