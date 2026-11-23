import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize("inplace", (False, True))
@pytest.mark.parametrize(*generate_arguments())
def test_relu(shape, inplace, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.relu(input, inplace)
    reference_output = F.relu(input, inplace)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
