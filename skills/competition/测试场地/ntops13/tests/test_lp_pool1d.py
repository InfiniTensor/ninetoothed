import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("norm_type", (1.0, 2.0, 3.0))
@pytest.mark.parametrize("ceil_mode", (False, True))
@pytest.mark.parametrize("use_stride", (False, True))
def test_lp_pool1d(norm_type, ceil_mode, use_stride):
    device = "cuda"
    dtype = torch.float32

    batch = random.randint(1, 4)
    channels = random.randint(1, 4)
    length = random.randint(4, 32)

    kernel_size = random.randint(1, min(5, length))

    if use_stride:
        stride = random.randint(1, max(1, kernel_size))
    else:
        stride = None

    input_tensor = torch.randn((batch, channels, length), device=device, dtype=dtype)

    ntops_output = ntops.torch.lp_pool1d(
        input_tensor,
        norm_type,
        kernel_size,
        stride=stride,
        ceil_mode=ceil_mode,
    )
    reference_output = torch.nn.functional.lp_pool1d(
        input_tensor,
        norm_type,
        kernel_size,
        stride=stride,
        ceil_mode=ceil_mode,
    )

    assert torch.allclose(ntops_output, reference_output, atol=1e-3, rtol=1e-3, equal_nan=True)
