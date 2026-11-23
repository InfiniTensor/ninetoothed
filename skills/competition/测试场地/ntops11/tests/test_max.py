import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
@pytest.mark.parametrize("keepdim", (False, True))
def test_max_dim(shape, dtype, device, rtol, atol, keepdim):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    dim = random.randint(0, input_tensor.ndim - 1)

    if random.choice((True, False)):
        dim = dim - input_tensor.ndim

    ntops_values, ntops_indices = ntops.torch.max(input_tensor, dim=dim, keepdim=keepdim)
    reference_values, reference_indices = torch.max(input_tensor, dim=dim, keepdim=keepdim)

    assert torch.allclose(ntops_values, reference_values, rtol=rtol, atol=atol)
    assert torch.equal(ntops_indices, reference_indices)


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_max_global(shape, dtype, device, rtol, atol):
    """测试全局 max (dim=None)，返回标量最大值"""
    input_tensor = torch.randn(shape, dtype=dtype, device=device)

    ntops_value = ntops.torch.max(input_tensor)
    reference_value = torch.max(input_tensor)

    assert torch.allclose(ntops_value, reference_value, rtol=rtol, atol=atol)
