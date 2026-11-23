import pytest
import torch
import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments

@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_maximum_elementwise(shape, dtype, device, rtol, atol):
    """测试基础的逐元素 maximum (形状相同)"""
    # 随机生成测试数据
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    other_tensor = torch.randn(shape, dtype=dtype, device=device)

    # 1. 运行你的 DSL 实现
    # 注意：确保 ntops.torch.maximum 已经正确暴露
    ntops_result = ntops.torch.maximum(input_tensor, other_tensor)
    
    # 2. 运行 PyTorch 参考实现
    reference_result = torch.maximum(input_tensor, other_tensor)

    # 3. 验证结果
    assert torch.allclose(ntops_result, reference_result, rtol=rtol, atol=atol)

@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_maximum_broadcasting(dtype):
    """测试广播机制 (例如: [4, 1, 32] vs [1, 64, 32])"""
    device = "cuda"
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    shape_a = (4, 1, 32)
    shape_b = (1, 64, 32)
    expected_shape = (4, 64, 32)
    
    input_tensor = torch.randn(shape_a, dtype=dtype, device=device)
    other_tensor = torch.randn(shape_b, dtype=dtype, device=device)

    ntops_result = ntops.torch.maximum(input_tensor, other_tensor)
    reference_result = torch.maximum(input_tensor, other_tensor)

    # 检查形状是否正确广播
    assert ntops_result.shape == expected_shape
    # 检查数值正确性
    assert torch.allclose(ntops_result, reference_result)

@skip_if_cuda_not_available
def test_maximum_out_variant():
    """测试 out= 参数"""
    device = "cuda"
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    x = torch.randn(100, device=device)
    y = torch.randn(100, device=device)
    out = torch.empty_like(x)
    
    ntops.torch.maximum(x, y, out=out)
    expected = torch.maximum(x, y)
    
    assert torch.allclose(out, expected)