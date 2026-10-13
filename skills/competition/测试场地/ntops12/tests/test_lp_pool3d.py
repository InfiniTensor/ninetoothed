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
def test_lp_pool3d(norm_type, ceil_mode, use_tuple_kernel, use_stride):
    device = "cuda"
    dtype = torch.float32

    batch = random.randint(1, 2)
    channels = random.randint(1, 3)
    depth = random.randint(4, 16)
    height = random.randint(4, 16)
    width = random.randint(4, 16)

    if use_tuple_kernel:
        k_d = random.randint(1, min(4, depth))
        k_h = random.randint(1, min(4, height))
        k_w = random.randint(1, min(4, width))
        kernel_size = (k_d, k_h, k_w)
    else:
        k = random.randint(1, min(4, depth, height, width))
        kernel_size = k

    if use_stride:
        if use_tuple_kernel and isinstance(kernel_size, tuple):
            s_d = random.randint(1, max(1, kernel_size[0]))
            s_h = random.randint(1, max(1, kernel_size[1]))
            s_w = random.randint(1, max(1, kernel_size[2]))
            stride = (s_d, s_h, s_w)
        else:
            base = kernel_size if isinstance(kernel_size, int) else min(kernel_size)
            stride = random.randint(1, max(1, base))
    else:
        stride = None

    input_tensor = torch.randn((batch, channels, depth, height, width), device=device, dtype=dtype)

    ntops_output = ntops.torch.lp_pool3d(
        input_tensor,
        norm_type,
        kernel_size,
        stride=stride,
        ceil_mode=ceil_mode,
    )
    reference_output = torch.nn.functional.lp_pool3d(
        input_tensor,
        norm_type,
        kernel_size,
        stride=stride,
        ceil_mode=ceil_mode,
    )

    assert torch.allclose(ntops_output, reference_output, atol=1e-3, rtol=1e-3, equal_nan=True)
