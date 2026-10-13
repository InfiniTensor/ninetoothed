import random
import pytest
import torch
import ntops
from tests.skippers import skip_if_cuda_not_available

@skip_if_cuda_not_available
@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("use_tuple", [False, True])
def test_max_pool3d(ceil_mode, padding, use_tuple):
    device = "cuda"
    dtype = torch.float32
    
    batch = random.randint(1, 2)
    channels = random.randint(1, 3)
    depth = random.randint(8, 16)
    height = random.randint(8, 16)
    width = random.randint(8, 16)

    if use_tuple:
        kernel_size = (random.randint(2, 3), random.randint(2, 3), random.randint(2, 3))
        stride = (random.randint(1, 2), random.randint(1, 2), random.randint(1, 2))
    else:
        kernel_size = random.randint(2, 3)
        stride = random.randint(1, 2)

    input_tensor = torch.randn((batch, channels, depth, height, width), device=device, dtype=dtype)

    # Ntops implementation
    ntops_output = ntops.torch.max_pool3d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode
    )

    # Reference implementation
    reference_output = torch.nn.functional.max_pool3d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode
    )

    assert torch.allclose(ntops_output, reference_output, atol=1e-3, rtol=1e-3)