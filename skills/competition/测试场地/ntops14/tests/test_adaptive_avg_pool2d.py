import random
import pytest
import torch
import ntops
from tests.skippers import skip_if_cuda_not_available

@skip_if_cuda_not_available
@pytest.mark.parametrize("output_size", [(1, 1), (2, 2), (4, 4)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_adaptive_avg_pool2d(output_size, dtype):
    device = "cuda"

    batch = random.randint(1, 3)
    channels = random.randint(1, 4)
    
    # 构造能整除的 Shape 以保证与 PyTorch 行为一致
    # 因为 DSL 使用固定 stride/kernel，而 PyTorch 在不能整除时使用变长 stride
    base_h, base_w = output_size
    height = base_h * random.randint(2, 6)
    width = base_w * random.randint(2, 6)

    input_tensor = torch.randn((batch, channels, height, width), device=device, dtype=dtype)

    # Ntops implementation
    ntops_output = ntops.torch.adaptive_avg_pool2d(
        input_tensor,
        output_size
    )

    # Reference implementation
    reference_output = torch.nn.functional.adaptive_avg_pool2d(
        input_tensor,
        output_size
    )

    # float16 容差稍大
    atol = 1e-3 if dtype == torch.float32 else 5e-3
    rtol = 1e-3 if dtype == torch.float32 else 5e-3
    
    assert torch.allclose(ntops_output, reference_output, atol=atol, rtol=rtol)