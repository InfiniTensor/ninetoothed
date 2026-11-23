import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_clamp(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    min = torch.randn(shape, dtype=dtype, device=device)
    max = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.clamp(input, min, max)
    reference_output = torch.clamp(input, min, max)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
