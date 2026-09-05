import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize("k", (0, 1, 2, 3))
@pytest.mark.parametrize(*generate_arguments())
def test_rot90(shape, dtype, device, rtol, atol, k):
    if len(shape) < 2:
        shape.append(2)

    input = torch.randn(shape, dtype=dtype, device=device)
    k += random.randint(-100, 100) * 4

    dim_0 = random.randint(0, len(shape) - 1)
    dim_1 = random.randint(0, len(shape) - 1)

    if dim_0 == dim_1:
        dim_1 = (dim_1 + 1) % len(shape)

    dims = (dim_0, dim_1)

    ninetoothed_output = ntops.torch.rot90(input, k=k, dims=dims)
    reference_output = torch.rot90(input, k=k, dims=dims)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
