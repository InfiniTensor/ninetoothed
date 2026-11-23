import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_ge(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.ge(input, other)
    reference_output = torch.ge(input, other)

    assert torch.equal(ninetoothed_output, reference_output)
