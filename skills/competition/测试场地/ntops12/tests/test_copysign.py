import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_copysign(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.copysign(input, other)
    reference_output = torch.copysign(input, other)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_copysign_special_values(dtype):
    device = "cuda"
    input = torch.tensor([0.0, -0.0, 1.5, -2.5, float("nan"), float("inf"), float("-inf")], dtype=dtype, device=device)
    other = torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0], dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.copysign(input, other)
    reference_output = torch.copysign(input, other)

    ninetoothed_bits = ninetoothed_output.view(torch.int16 if dtype == torch.float16 else torch.int32)
    reference_bits = reference_output.view(torch.int16 if dtype == torch.float16 else torch.int32)

    assert torch.equal(ninetoothed_bits, reference_bits)
