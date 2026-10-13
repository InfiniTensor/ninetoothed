import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize("eps", (None, 0, 1e-5, 1e-3))
@pytest.mark.parametrize("weight_is_none", (False, True))
@pytest.mark.parametrize(*generate_arguments())
def test_rms_norm(shape, dtype, device, rtol, atol, weight_is_none, eps):
    input = torch.randn(shape, dtype=dtype, device=device)
    normalized_shape = shape[-random.randint(1, len(shape)) :]
    if weight_is_none:
        weight = None
    else:
        weight = torch.randn(normalized_shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.rms_norm(
        input, normalized_shape, weight=weight, eps=eps
    )
    reference_output = torch.nn.functional.rms_norm(
        input, normalized_shape, weight=weight, eps=eps
    )

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
