import random

import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_threshold(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    threshold = random.uniform(-1, 1)
    value = random.uniform(0, 1)

    ninetoothed_output = ntops.torch.threshold(input, threshold, value)
    reference_output = F.threshold(input, threshold, value)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
