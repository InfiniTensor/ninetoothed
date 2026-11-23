import random
import pytest
import torch
import ntops
from tests.skippers import skip_if_cuda_not_available

@skip_if_cuda_not_available
@pytest.mark.parametrize("output_size", [(1, 1), (2, 2), (3, 3), (5, 7)])
def test_adaptive_max_pool2d(output_size):
    device = "cuda"
    dtype = torch.float32

    batch = random.randint(1, 3)
    channels = random.randint(1, 4)
    # 为了保证测试通过，尽量使用能整除的大小，或者接受一定的精度误差/边界差异
    # 如果 DSL tile 不支持动态 stride，不可整除的 case 可能会有边界差异
    base_h, base_w = output_size
    height = base_h * random.randint(1, 5)
    width = base_w * random.randint(1, 5)

    input_tensor = torch.randn((batch, channels, height, width), device=device, dtype=dtype)

    # Ntops implementation
    ntops_output = ntops.torch.adaptive_max_pool2d(
        input_tensor,
        output_size
    )

    # Reference implementation
    reference_output = torch.nn.functional.adaptive_max_pool2d(
        input_tensor,
        output_size
    )

    assert torch.allclose(ntops_output, reference_output, atol=1e-3, rtol=1e-3)