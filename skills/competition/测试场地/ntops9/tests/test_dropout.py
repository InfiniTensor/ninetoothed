import random

import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_dropout(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    p = random.uniform(0, 1)

    # TODO: Add `training` and `inplace` tests later.
    ninetoothed_output = ntops.torch.dropout(input, p=p)
    reference_output = F.dropout(input, p=p)

    assert ninetoothed_output.shape == reference_output.shape

    ninetoothed_non_zero_ratio = (
        ninetoothed_output.nonzero().numel() / ninetoothed_output.ndim / input.numel()
    )
    reference_non_zero_ratio = (
        reference_output.nonzero().numel() / reference_output.ndim / input.numel()
    )

    assert abs(ninetoothed_non_zero_ratio - reference_non_zero_ratio) < 0.1

    assert torch.allclose(
        ninetoothed_output[ninetoothed_output != 0],
        input[ninetoothed_output != 0] / (1 - p),
    )
