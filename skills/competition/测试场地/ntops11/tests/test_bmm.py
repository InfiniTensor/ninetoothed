import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.test_mm import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_bmm(m, n, k, dtype, device, rtol, atol):
    b = random.randint(4, 16)
    input = torch.randn((b, m, k), dtype=dtype, device=device)
    other = torch.randn((b, k, n), dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.bmm(input, other)
    reference_output = torch.bmm(input, other)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
