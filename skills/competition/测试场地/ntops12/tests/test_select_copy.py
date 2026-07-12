import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_select_copy(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    dim = random.randint(0, input.ndim - 1)
    index = random.randint(0, input.size(dim) - 1)

    ninetoothed_output = ntops.torch.select_copy(input, dim=dim, index=index)
    reference_output = torch.select_copy(input, dim=dim, index=index)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
