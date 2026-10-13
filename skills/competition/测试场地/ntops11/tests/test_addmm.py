import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.test_mm import generate_arguments
from tests.utils import gauss


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_addmm(m, n, k, dtype, device, rtol, atol):
    input = torch.randn((m, n), dtype=dtype, device=device)
    x = torch.randn((m, k), dtype=dtype, device=device)
    y = torch.randn((k, n), dtype=dtype, device=device)
    beta = gauss()
    alpha = gauss()

    ninetoothed_output = ntops.torch.addmm(input, x, y, beta=beta, alpha=alpha)
    reference_output = torch.addmm(input, x, y, beta=beta, alpha=alpha)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
