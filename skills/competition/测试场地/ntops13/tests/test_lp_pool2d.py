import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("norm_type", (1.0, 2.0, 3.0))
@pytest.mark.parametrize("ceil_mode", (False, True))
@pytest.mark.parametrize("use_tuple_kernel", (False, True))
@pytest.mark.parametrize("use_stride", (False, True))
def test_lp_pool2d(norm_type, ceil_mode, use_tuple_kernel, use_stride):
    device = "cuda"
    dtype = torch.float32

    batch = random.randint(1, 3)
    channels = random.randint(1, 4)
    height = random.randint(4, 24)
    width = random.randint(4, 24)

    if use_tuple_kernel:
        k_h = random.randint(1, min(5, height))
        k_w = random.randint(1, min(5, width))
        kernel_size = (k_h, k_w)
    else:
        k = random.randint(1, min(5, height, width))
        kernel_size = k

    if use_stride:
        if use_tuple_kernel and isinstance(kernel_size, tuple):
            s_h = random.randint(1, max(1, kernel_size[0]))
            s_w = random.randint(1, max(1, kernel_size[1]))
            stride = (s_h, s_w)
        else:
            stride = random.randint(1, max(1, kernel_size if isinstance(kernel_size, int) else min(kernel_size)))
    else:
        stride = None

    input_tensor = torch.randn((batch, channels, height, width), device=device, dtype=dtype)

    ntops_output = ntops.torch.lp_pool2d(
        input_tensor,
        norm_type,
        kernel_size,
        stride=stride,
        ceil_mode=ceil_mode,
    )
    reference_output = torch.nn.functional.lp_pool2d(
        input_tensor,
        norm_type,
        kernel_size,
        stride=stride,
        ceil_mode=ceil_mode,
    )

    assert torch.allclose(ntops_output, reference_output, atol=1e-3, rtol=1e-3, equal_nan=True)
