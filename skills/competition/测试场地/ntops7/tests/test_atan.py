import pytest
import torch
import math

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_atan_basic(shape, dtype, device, rtol, atol):
    """
    基本功能测试：对比 PyTorch 原生实现
    """
    # atan 定义域是全实数，生成范围在 [-10, 10] 的随机数
    input_tensor = (torch.rand(shape, dtype=dtype, device=device) - 0.5) * 20.0
    
    reference_output = torch.atan(input_tensor)
    ntops_output = ntops.torch.atan(input_tensor)
    
    assert torch.allclose(ntops_output, reference_output, rtol=rtol, atol=atol)
    assert ntops_output.shape == input_tensor.shape
    assert ntops_output.dtype == input_tensor.dtype


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cuda"])
def test_atan_boundary_values(dtype, device):
    """
    边界值测试：0, 1, -1, inf, -inf
    """
    test_values = torch.tensor([
        0.0, 
        1.0, 
        -1.0, 
        float('inf'), 
        -float('inf')
    ], dtype=dtype, device=device)
    
    reference_output = torch.atan(test_values)
    ntops_output = ntops.torch.atan(test_values)
    
    # 验证数值精度
    # float16 精度较低，适当放宽误差
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    
    assert torch.allclose(ntops_output, reference_output, rtol=rtol, atol=atol)
    
    # 显式验证特殊数学性质
    # atan(0) = 0
    assert torch.abs(ntops_output[0]) < 1e-6
    # atan(inf) = pi/2
    assert torch.abs(ntops_output[3] - (math.pi / 2)) < (1e-3 if dtype == torch.float16 else 1e-6)
    # atan(-inf) = -pi/2
    assert torch.abs(ntops_output[4] - (-math.pi / 2)) < (1e-3 if dtype == torch.float16 else 1e-6)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_atan_strided_output(dtype):
    """
    测试非连续内存布局 (Strided Output)
    """
    device = "cuda"
    input_tensor = torch.randn(4, 5, 6, dtype=dtype, device=device)
    reference_output = torch.atan(input_tensor)
    
    # 创建非连续输出张量
    large_tensor = torch.empty(4, 5, 8, dtype=dtype, device=device)
    out = large_tensor[:, :, :6]
    
    result = ntops.torch.atan(input_tensor, out=out)
    
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    
    assert torch.allclose(result, reference_output, rtol=rtol, atol=atol)
    assert result is out
    assert not out.is_contiguous()


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_atan_inplace_compatibility(dtype):
    """
    测试与原地操作的兼容性
    """
    device = "cuda"
    input_tensor = torch.randn(10, 10, dtype=dtype, device=device)
    input_copy = input_tensor.clone()
    
    # 链式操作
    input_copy.mul_(2.0)
    input_copy.add_(0.5)
    
    reference_output = torch.atan(input_copy)
    ntops_output = ntops.torch.atan(input_copy)
    
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    
    assert torch.allclose(ntops_output, reference_output, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_atan_nan_handling(dtype):
    """
    测试 NaN 输入处理
    """
    device = "cuda"
    input_tensor = torch.tensor([float('nan'), 1.0], dtype=dtype, device=device)
    
    output = ntops.torch.atan(input_tensor)
    
    assert torch.isnan(output[0])
    assert not torch.isnan(output[1])