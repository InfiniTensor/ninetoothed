import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.test_mm import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
@pytest.mark.parametrize("b", (None, 1, 2, 3))
def test_matmul(b, m, n, k, dtype, device, rtol, atol):
    input_shape = (b, m, k) if b is not None else (m, k)
    other_shape = (b, k, n) if b is not None else (k, n)

    input = torch.randn(input_shape, dtype=dtype, device=device)
    other = torch.randn(other_shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.matmul(input, other)
    reference_output = torch.matmul(input, other)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
