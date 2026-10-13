import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize("stable", (False, True))
@pytest.mark.parametrize("descending", (False, True))
@pytest.mark.parametrize(*generate_arguments())
def test_sort(shape, dtype, device, rtol, atol, descending, stable):
    input = torch.randn(shape, dtype=dtype, device=device)
    dim = random.randint(-input.ndim, input.ndim - 1)

    ninetoothed_output = ntops.torch.sort(
        input, dim=dim, descending=descending, stable=stable
    )
    reference_output = torch.sort(input, dim=dim, descending=descending, stable=stable)

    assert torch.allclose(
        ninetoothed_output.values, reference_output.values, rtol=rtol, atol=atol
    )
    assert torch.equal(ninetoothed_output.indices, reference_output.indices)


@skip_if_cuda_not_available
@pytest.mark.parametrize("descending", (False, True))
def test_sort_stable_with_duplicate_values(descending):
    input = torch.randint(-4, 5, (16, 33), dtype=torch.int32, device="cuda").to(
        torch.float32
    )

    ninetoothed_output = ntops.torch.sort(
        input, dim=-1, descending=descending, stable=True
    )
    reference_output = torch.sort(input, dim=-1, descending=descending, stable=True)

    assert torch.equal(ninetoothed_output.indices, reference_output.indices)
    assert torch.equal(ninetoothed_output.values, reference_output.values)


@skip_if_cuda_not_available
@pytest.mark.parametrize("descending", (False, True))
def test_sort_with_out(descending):
    input = torch.randn((19, 23), dtype=torch.float16, device="cuda")
    values = torch.empty_like(input)
    indices = torch.empty_like(input, dtype=torch.int64)

    ninetoothed_output = ntops.torch.sort(
        input, dim=-1, descending=descending, out=(values, indices)
    )
    reference_output = torch.sort(input, dim=-1, descending=descending)

    assert ninetoothed_output.values.data_ptr() == values.data_ptr()
    assert ninetoothed_output.indices.data_ptr() == indices.data_ptr()
    assert torch.equal(values, reference_output.values)
    assert torch.equal(indices, reference_output.indices)
