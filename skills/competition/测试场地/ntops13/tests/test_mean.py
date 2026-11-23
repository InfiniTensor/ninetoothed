import random
import pytest
import torch
import ntops
# 假设你在 ntops 包里导出了 mean
from ntops.torch.mean import mean as ntops_mean 

from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments

@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
@pytest.mark.parametrize("keepdim", (False, True))
def test_mean_dim(shape, dtype, device, rtol, atol, keepdim):
    # Mean 测试通常需要稍微放宽 float16 的误差容忍度，因为除法会引入额外误差
    if dtype == torch.float16:
        atol = max(atol, 1e-3)
        rtol = max(rtol, 1e-3)

    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    dim = random.randint(0, input_tensor.ndim - 1)

    if random.choice((True, False)):
        dim = dim - input_tensor.ndim

    ntops_value = ntops_mean(input_tensor, dim=dim, keepdim=keepdim)
    reference_value = torch.mean(input_tensor, dim=dim, keepdim=keepdim)

    assert torch.allclose(ntops_value, reference_value, rtol=rtol, atol=atol)

@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_mean_global(shape, dtype, device, rtol, atol):
    if dtype == torch.float16:
        atol = max(atol, 1e-3)
        rtol = max(rtol, 1e-3)

    input_tensor = torch.randn(shape, dtype=dtype, device=device)

    ntops_value = ntops_mean(input_tensor)
    reference_value = torch.mean(input_tensor)

    assert torch.allclose(ntops_value, reference_value, rtol=rtol, atol=atol)

@skip_if_cuda_not_available
def test_mean_int_input():
    # 特别测试：整数输入应该产生浮点输出
    device = "cuda"
    shape = (1024, 1024)
    input_tensor = torch.randint(0, 10, shape, device=device, dtype=torch.int32)
    
    ntops_value = ntops_mean(input_tensor)
    reference_value = torch.mean(input_tensor.float()) # torch.mean(int)在旧版本可能报错或行为不同，通常需转float
    
    assert ntops_value.is_floating_point()
    assert torch.allclose(ntops_value, reference_value)