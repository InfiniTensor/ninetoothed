import math
import random

import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments

_ALPHA_P = -1.7580993408473766


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_alpha_dropout(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    p = random.uniform(0.1, 0.5)

    ninetoothed_output = ntops.torch.alpha_dropout(input, p=p, training=True)
    reference_output = F.alpha_dropout(input, p=p, training=True)

    # 1. Shape must match.
    assert ninetoothed_output.shape == reference_output.shape

    # 2. Compute expected affine parameters.
    q = 1.0 - p
    a = 1.0 / math.sqrt(q * (1.0 + p * _ALPHA_P * _ALPHA_P))
    b = -a * p * _ALPHA_P
    sat = a * _ALPHA_P + b

    # 3. Drop ratios should be close to each other.
    ninetoothed_drop_ratio = (
        torch.isclose(
            ninetoothed_output, torch.full_like(ninetoothed_output, sat), atol=atol
        )
        .float()
        .mean()
        .item()
    )
    reference_drop_ratio = (
        torch.isclose(
            reference_output, torch.full_like(reference_output, sat), atol=atol
        )
        .float()
        .mean()
        .item()
    )

    assert abs(ninetoothed_drop_ratio - reference_drop_ratio) < 0.1

    # 4. Kept elements should satisfy the same affine transform.
    kept_mask = ~torch.isclose(
        ninetoothed_output, torch.full_like(ninetoothed_output, sat), atol=atol
    )
    expected_kept = a * input[kept_mask].float() + b
    actual_kept = ninetoothed_output[kept_mask].float()

    assert torch.allclose(actual_kept, expected_kept, rtol=rtol, atol=atol)

    # 5. training=False should return input unchanged.
    output_eval = ntops.torch.alpha_dropout(input, p=p, training=False)
    assert torch.equal(output_eval, input)
