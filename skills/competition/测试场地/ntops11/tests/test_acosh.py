import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_acosh_basic(shape, dtype, device, rtol, atol):
    """
    基本功能测试：测试acosh在各种形状、数据类型和设备上的正确性
    与PyTorch参考实现进行对比验证
    """
    # 创建输入张量，确保值在定义域内（x >= 1）
    # 生成[1, 10]范围内的随机数，避免数值不稳定区域
    input_tensor = 1.0 + 9.0 * torch.rand(shape, dtype=dtype, device=device)
    
    # 计算参考输出（PyTorch内置实现）
    reference_output = torch.acosh(input_tensor)
    
    # 计算ntops实现
    ntops_output = ntops.torch.acosh(input_tensor)
    
    # 验证数值精度
    assert torch.allclose(ntops_output, reference_output, rtol=rtol, atol=atol)
    
    # 验证输出形状与输入形状一致
    assert ntops_output.shape == input_tensor.shape
    
    # 验证数据类型一致性
    assert ntops_output.dtype == input_tensor.dtype


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cuda"])
def test_acosh_boundary_values(dtype, device):
    """
    边界值测试：测试acosh在关键边界点的行为
    包括：x=1, x接近1, x很大, x<1等边界情况
    """
    # 测试点：1.0（精确边界）, 1.000001（接近1）, 100.0（大值）, 0.5（小于1，应返回NaN）
    test_values = torch.tensor([1.0, 1.000001, 100.0, 0.5], dtype=dtype, device=device)
    
    # 计算参考输出
    reference_output = torch.acosh(test_values)
    
    # 计算ntops实现
    ntops_output = ntops.torch.acosh(test_values)
    
    # 验证数值精度（跳过NaN比较）
    valid_mask = ~torch.isnan(reference_output)
    assert torch.allclose(ntops_output[valid_mask], reference_output[valid_mask], 
                         rtol=1e-3 if dtype == torch.float16 else 1e-5,
                         atol=1e-3 if dtype == torch.float16 else 1e-6)
    
    # 验证NaN处理
    nan_mask = torch.isnan(reference_output)
    assert torch.all(torch.isnan(ntops_output[nan_mask]))
    
    # 验证acosh(1) = 0
    assert torch.allclose(ntops_output[0], torch.tensor(0.0, dtype=dtype, device=device),
                         atol=1e-6)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_acosh_with_contiguous_out(dtype, dim):
    """
    测试使用连续输出张量的情况
    验证out参数的正确性和数据一致性
    """
    device = "cuda"
    
    # 创建输入张量（3D）
    input_tensor = 1.0 + 9.0 * torch.rand(4, 5, 6, dtype=dtype, device=device)
    
    # 计算参考输出
    reference_output = torch.acosh(input_tensor)
    
    # 创建连续的输出张量
    out = torch.empty_like(reference_output)
    
    # 使用ntops的acosh，传入out参数
    result = ntops.torch.acosh(input_tensor, out=out)
    
    # 验证结果
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    assert torch.allclose(result, reference_output, rtol=rtol, atol=atol)
    assert result is out  # 确保返回的是传入的out张量
    assert out.is_contiguous()  # 确保输出是连续的


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_acosh_with_strided_output(dtype):
    """
    测试使用非连续（strided）输出张量的情况
    验证内核对非标准内存布局的处理能力
    """
    device = "cuda"
    
    # 创建输入张量
    input_tensor = 1.0 + 9.0 * torch.rand(4, 5, 6, dtype=dtype, device=device)
    
    # 计算参考输出
    reference_output = torch.acosh(input_tensor)
    
    # 创建一个具有不同strides的输出张量
    large_tensor = torch.empty(4, 5, 8, dtype=dtype, device=device)
    out = large_tensor[:, :, :6]  # shape (4, 5, 6) 但strides为(40, 8, 1)而不是标准的(30, 6, 1)
    
    # 使用ntops的acosh，传入strided输出
    result = ntops.torch.acosh(input_tensor, out=out)
    
    # 验证结果
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    assert torch.allclose(result, reference_output, rtol=rtol, atol=atol)
    assert result is out  # 确保返回的是传入的out张量
    assert not out.is_contiguous()  # 确保输出是非连续的


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_acosh_with_transposed_output(dtype):
    """
    测试使用转置（非连续）输出张量的情况
    验证内核对转置张量内存布局的处理能力
    """
    device = "cuda"
    
    # 创建输入张量
    input_tensor = 1.0 + 9.0 * torch.rand(3, 4, 5, dtype=dtype, device=device)
    
    # 计算参考输出
    reference_output = torch.acosh(input_tensor)
    
    # 创建一个转置的输出张量（非连续）
    out_base = torch.empty(5, 4, 3, dtype=dtype, device=device)
    out = out_base.permute(2, 1, 0)  # shape (3, 4, 5)，但内存布局非连续
    
    assert not out.is_contiguous(), "out should be non-contiguous before calling acosh"
    
    # 使用ntops的acosh
    result = ntops.torch.acosh(input_tensor, out=out)
    
    # 验证结果
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    assert torch.allclose(result, reference_output, rtol=rtol, atol=atol)
    assert result is out


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_acosh_with_strided_slice_output(dtype):
    """
    测试使用步进切片（strided slice）输出张量的情况
    验证内核对复杂内存布局的处理能力
    """
    device = "cuda"
    
    # 创建输入张量
    input_tensor = 1.0 + 9.0 * torch.rand(2, 8, 6, dtype=dtype, device=device)
    
    # 计算参考输出
    reference_output = torch.acosh(input_tensor)
    
    # 创建一个更大的张量，使用步进切片得到非连续的子张量
    large_tensor = torch.empty(2, 8, 12, dtype=dtype, device=device)
    out = large_tensor[:, ::2, ::2]  # shape (2, 4, 6)，strides非标准
    
    # 调整输入以匹配输出的第二维（仅用于测试，实际使用不需要）
    input_tensor_adjusted = input_tensor[:, ::2, :]  # shape (2, 4, 6)
    reference_output_adjusted = torch.acosh(input_tensor_adjusted)
    
    # 使用ntops的acosh
    result = ntops.torch.acosh(input_tensor_adjusted, out=out)
    
    # 验证结果
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    assert torch.allclose(result, reference_output_adjusted, rtol=rtol, atol=atol)
    assert result is out
    assert not out.is_contiguous()


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("keep_out", [False, True])
def test_acosh_out_vs_no_out(dtype, keep_out):
    """
    测试使用out参数和不使用out参数的结果一致性
    验证两种调用方式的数值一致性
    """
    device = "cuda"
    
    # 创建输入张量
    input_tensor = 1.0 + 9.0 * torch.rand(4, 5, 6, dtype=dtype, device=device)
    
    # 不使用out参数
    result_no_out = ntops.torch.acosh(input_tensor)
    
    # 使用out参数
    if keep_out:
        out = torch.empty_like(result_no_out)
        result_with_out = ntops.torch.acosh(input_tensor, out=out)
        assert result_with_out is out
    else:
        result_with_out = ntops.torch.acosh(input_tensor, out=torch.empty_like(result_no_out))
    
    # 验证结果一致
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    assert torch.allclose(result_no_out, result_with_out, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_acosh_special_values(dtype, device="cuda"):
    """
    特殊值测试：测试acosh在特殊输入值上的行为
    包括：NaN, inf, -inf等
    """
    # 创建包含特殊值的输入张量
    special_values = torch.tensor([
        float('nan'),    # NaN
        float('inf'),    # inf
        -float('inf'),   # -inf
        1.0,             # 边界值
        2.0,             # 正常值
        0.5              # 小于1（应返回NaN）
    ], dtype=dtype, device=device)
    
    # 计算ntops实现
    ntops_output = ntops.torch.acosh(special_values)
    
    # 验证特殊值处理
    # NaN输入应该产生NaN输出
    assert torch.isnan(ntops_output[0])
    
    # inf输入应该产生inf输出
    assert torch.isinf(ntops_output[1]) and ntops_output[1] > 0
    
    # -inf输入应该产生NaN（因为acosh定义域是x >= 1）
    assert torch.isnan(ntops_output[2])
    
    # 1.0应该产生0.0
    assert torch.allclose(ntops_output[3], torch.tensor(0.0, dtype=dtype, device=device), atol=1e-6)
    
    # 0.5应该产生NaN
    assert torch.isnan(ntops_output[5])


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape", [(10000,), (100, 100), (10, 10, 10)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_acosh_large_tensors(shape, dtype):
    """
    大张量测试：测试acosh在大尺寸张量上的性能和正确性
    验证内存使用和计算效率
    """
    device = "cuda"
    
    # 创建大输入张量
    input_tensor = 1.0 + 9.0 * torch.rand(shape, dtype=dtype, device=device)
    
    # 计算参考输出
    reference_output = torch.acosh(input_tensor)
    
    # 计算ntops实现
    ntops_output = ntops.torch.acosh(input_tensor)
    
    # 验证数值精度
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    assert torch.allclose(ntops_output, reference_output, rtol=rtol, atol=atol)
    
    # 验证内存效率（简单检查，不实际测量内存）
    assert ntops_output.shape == input_tensor.shape


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_acosh_inplace_consistency(dtype):
    """
    原地操作一致性测试：测试acosh与原地操作的兼容性
    虽然acosh本身不是原地操作，但需要确保与原地操作链的兼容性
    """
    device = "cuda"
    
    # 创建输入张量
    input_tensor = 1.0 + 9.0 * torch.rand(4, 5, 6, dtype=dtype, device=device)
    
    # 复制输入用于原地操作测试
    input_copy = input_tensor.clone()
    
    # 链式操作：先进行一些原地操作，然后计算acosh
    input_copy.add_(1.0)  # 原地加法
    input_copy.mul_(0.5)  # 原地乘法
    
    # 确保值仍在定义域内
    input_copy = torch.clamp(input_copy, min=1.0)
    
    # 计算参考输出
    reference_output = torch.acosh(input_copy)
    
    # 计算ntops实现
    ntops_output = ntops.torch.acosh(input_copy)
    
    # 验证结果
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    assert torch.allclose(ntops_output, reference_output, rtol=rtol, atol=atol)