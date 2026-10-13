import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_isinf(shape, dtype, device, rtol, atol):
    def generate_inf_tensor(shape, dtype, device):
        x = torch.randn(shape, dtype=dtype, device=device)

        probs = (0.2, 0.6)
        prob_tensor = torch.rand(shape, device=device)

        mask = (probs[0] < prob_tensor) & (prob_tensor < probs[1])
        x[mask] = float("inf")
        mask = probs[1] < prob_tensor
        x[mask] = float("-inf")

        return x

    input = generate_inf_tensor(shape, dtype, device)

    ninetoothed_output = ntops.torch.isinf(input)
    reference_output = torch.isinf(input)

    assert torch.equal(ninetoothed_output, reference_output)
