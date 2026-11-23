import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import gauss, generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_sub(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)
    alpha = gauss()

    ninetoothed_output = ntops.torch.sub(input, other, alpha=alpha)
    reference_output = torch.sub(input, other, alpha=alpha)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
