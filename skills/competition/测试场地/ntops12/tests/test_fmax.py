import random
import pytest
import torch
import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments

@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_fmax_elementwise(shape, dtype, device, rtol, atol):
    """测试基础的逐元素 fmax (形状相同)"""
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    other_tensor = torch.randn(shape, dtype=dtype, device=device)

    # ntops 实现
    ntops_result = ntops.torch.fmax(input_tensor, other_tensor)
    
    # PyTorch 参考实现 (torch.fmax 或 torch.maximum)
    # torch.fmax: 忽略 NaN
    # torch.maximum: 传播 NaN
    # 取决于 ntl.maximum 的具体行为，这里通常对比 torch.maximum
    reference_result = torch.maximum(input_tensor, other_tensor)

    assert torch.allclose(ntops_result, reference_result, rtol=rtol, atol=atol)

@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fmax_broadcasting(dtype):
    """测试广播机制"""
    device = "cuda"
    shape_a = (4, 1, 32)
    shape_b = (1, 64, 32)
    
    input_tensor = torch.randn(shape_a, dtype=dtype, device=device)
    other_tensor = torch.randn(shape_b, dtype=dtype, device=device)

    ntops_result = ntops.torch.fmax(input_tensor, other_tensor)
    reference_result = torch.maximum(input_tensor, other_tensor)

    assert ntops_result.shape == (4, 64, 32)
    assert torch.allclose(ntops_result, reference_result)