import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
@pytest.mark.parametrize("keepdim", (False, True))
def test_logsumexp(shape, dtype, device, rtol, atol, keepdim):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    dim = random.randint(0, input_tensor.ndim - 1)

    if random.choice((True, False)):
        dim = dim - input_tensor.ndim

    reference_output = torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim)
    ntops_output = ntops.torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim)

    assert torch.allclose(ntops_output, reference_output, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("keepdim", [False, True])
def test_logsumexp_with_strided_output(dtype, keepdim):
    """测试 logsumexp 使用非连续（strided）输出张量的情况"""
    device = "cuda"
    
    # 创建输入张量
    input_tensor = torch.randn(4, 5, 6, dtype=dtype, device=device)
    dim = 2
    
    # 计算参考输出
    reference_output = torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim)
    
    # 创建一个具有不同 strides 的输出张量
    if keepdim:
        # 创建一个更大的张量，然后切片得到具有非标准 strides 的子张量
        large_tensor = torch.empty(4, 5, 3, dtype=dtype, device=device)
        out = large_tensor[:, :, :1]  # shape (4, 5, 1) 但 strides 为 (15, 3, 1) 而不是标准的 (5, 1, 1)
    else:
        # 创建一个更大的张量，然后切片得到具有非标准 strides 的子张量
        large_tensor = torch.empty(4, 5, 2, dtype=dtype, device=device)
        out = large_tensor[:, :, 0]  # shape (4, 5) 但 strides 为 (10, 2) 而不是标准的 (5, 1)
    
    # 使用 ntops 的 logsumexp，传入 strided 输出
    result = ntops.torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim, out=out)
    
    # 验证结果
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    assert torch.allclose(result, reference_output, rtol=rtol, atol=atol)
    assert result is out  # 确保返回的是传入的 out 张量


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_logsumexp_with_transposed_output(dtype):
    """测试 logsumexp 使用转置（非连续）输出张量的情况"""
    device = "cuda"
    
    # 创建输入张量
    input_tensor = torch.randn(3, 4, 5, dtype=dtype, device=device)
    dim = 1
    keepdim = True
    
    # 计算参考输出
    reference_output = torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim)
    
    # 创建一个转置的输出张量（非连续）
    out_base = torch.empty(1, 3, 5, dtype=dtype, device=device)
    out = out_base.transpose(0, 1)  # shape (3, 1, 5)，但内存布局非连续
    
    # 使用 ntops 的 logsumexp
    result = ntops.torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim, out=out)
    
    # 验证结果
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    assert torch.allclose(result, reference_output, rtol=rtol, atol=atol)
    assert result is out


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_logsumexp_with_strided_slice_output(dtype):
    """测试 logsumexp 使用步进切片（strided slice）输出张量的情况"""
    device = "cuda"
    
    # 创建输入张量
    input_tensor = torch.randn(2, 8, 6, dtype=dtype, device=device)
    dim = 2
    keepdim = True
    
    # 计算参考输出
    reference_output = torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim)
    
    # 创建一个更大的张量，使用步进切片得到非连续的子张量
    large_tensor = torch.empty(2, 8, 4, dtype=dtype, device=device)
    out = large_tensor[:, ::2, :1]  # shape (2, 4, 1)，strides 非标准
    
    # 调整输入以匹配输出的第二维
    input_tensor_adjusted = input_tensor[:, ::2, :]  # shape (2, 4, 6)
    reference_output_adjusted = torch.logsumexp(input_tensor_adjusted, dim=dim, keepdim=keepdim)
    
    # 使用 ntops 的 logsumexp
    result = ntops.torch.logsumexp(input_tensor_adjusted, dim=dim, keepdim=keepdim, out=out)
    
    # 验证结果
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    assert torch.allclose(result, reference_output_adjusted, rtol=rtol, atol=atol)
    assert result is out


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_logsumexp_with_contiguous_out(dtype, keepdim, dim):
    """测试 logsumexp 使用正常的连续（contiguous）输出张量"""
    device = "cuda"
    
    # 创建输入张量
    input_tensor = torch.randn(4, 5, 6, dtype=dtype, device=device)
    
    # 计算参考输出
    reference_output = torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim)
    
    # 创建一个连续的输出张量
    out = torch.empty_like(reference_output)
    
    # 使用 ntops 的 logsumexp，传入连续输出
    result = ntops.torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim, out=out)
    
    # 验证结果
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    assert torch.allclose(result, reference_output, rtol=rtol, atol=atol)
    assert result is out  # 确保返回的是传入的 out 张量
    assert out.is_contiguous()  # 确保输出是连续的


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("keepdim", [False, True])
def test_logsumexp_with_out_different_strides_dim0(dtype, keepdim):
    """测试 logsumexp 在 dim=0 时使用 strided 输出张量"""
    device = "cuda"
    
    # 创建输入张量
    input_tensor = torch.randn(6, 4, 5, dtype=dtype, device=device)
    dim = 0
    
    # 计算参考输出
    reference_output = torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim)
    
    # 创建一个具有不同 strides 的输出张量
    if keepdim:
        large_tensor = torch.empty(3, 4, 5, dtype=dtype, device=device)
        out = large_tensor[:1, :, :]  # shape (1, 4, 5) 但 strides 为 (20, 5, 1) 而不是标准的 (20, 5, 1)
    else:
        large_tensor = torch.empty(4, 5, 2, dtype=dtype, device=device)
        out = large_tensor[:, :, 0]  # shape (4, 5) 但 strides 为 (10, 2) 而不是标准的 (5, 1)
    
    # 使用 ntops 的 logsumexp
    result = ntops.torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim, out=out)
    
    # 验证结果
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    assert torch.allclose(result, reference_output, rtol=rtol, atol=atol)
    assert result is out


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("keepdim", [False, True])
def test_logsumexp_with_out_different_strides_dim1(dtype, keepdim):
    """测试 logsumexp 在 dim=1 时使用 strided 输出张量"""
    device = "cuda"
    
    # 创建输入张量
    input_tensor = torch.randn(3, 8, 5, dtype=dtype, device=device)
    dim = 1
    
    # 计算参考输出
    reference_output = torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim)
    
    # 创建一个具有不同 strides 的输出张量
    if keepdim:
        large_tensor = torch.empty(3, 2, 5, dtype=dtype, device=device)
        out = large_tensor[:, :1, :]  # shape (3, 1, 5) 但 strides 为 (10, 5, 1) 而不是标准的 (5, 5, 1)
    else:
        large_tensor = torch.empty(3, 5, 3, dtype=dtype, device=device)
        out = large_tensor[:, :, 0]  # shape (3, 5) 但 strides 为 (15, 3) 而不是标准的 (5, 1)
    
    # 使用 ntops 的 logsumexp
    result = ntops.torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim, out=out)
    
    # 验证结果
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    assert torch.allclose(result, reference_output, rtol=rtol, atol=atol)
    assert result is out


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_logsumexp_with_out_permuted(dtype):
    """测试 logsumexp 使用 permute 后的输出张量（非连续）"""
    device = "cuda"
    
    # 创建输入张量
    input_tensor = torch.randn(3, 4, 5, 6, dtype=dtype, device=device)
    dim = 2
    keepdim = True
    
    # 计算参考输出
    reference_output = torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim)
    
    # 创建一个 permute 后的输出张量（非连续）
    # 使用 permute(3, 2, 1, 0) 来确保输出是非连续的
    out_base = torch.empty(6, 1, 4, 3, dtype=dtype, device=device)
    out = out_base.permute(3, 2, 1, 0)  # shape (3, 4, 1, 6)，内存布局非连续
    
    assert not out.is_contiguous(), "out should be non-contiguous before calling logsumexp"
    
    # 使用 ntops 的 logsumexp
    result = ntops.torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim, out=out)
    
    # 验证结果
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    assert torch.allclose(result, reference_output, rtol=rtol, atol=atol)
    assert result is out


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_logsumexp_with_out_multiple_strides(dtype):
    """测试 logsumexp 使用多个维度都有非标准 strides 的输出张量"""
    device = "cuda"
    
    # 创建输入张量
    input_tensor = torch.randn(2, 6, 8, dtype=dtype, device=device)
    dim = 2
    keepdim = True
    
    # 计算参考输出
    reference_output = torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim)
    
    # 创建一个多个维度都有非标准 strides 的输出张量
    # 使用步进切片在多个维度上创建非连续张量
    large_tensor = torch.empty(4, 12, 3, dtype=dtype, device=device)
    out = large_tensor[::2, ::2, :1]  # shape (2, 6, 1)，所有维度的 strides 都非标准
    
    # 使用 ntops 的 logsumexp
    result = ntops.torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim, out=out)
    
    # 验证结果
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    assert torch.allclose(result, reference_output, rtol=rtol, atol=atol)
    assert result is out
    assert not out.is_contiguous()


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("keepdim", [False, True])
def test_logsumexp_out_vs_no_out(dtype, keepdim):
    """测试使用 out 参数和不使用 out 参数的结果一致性"""
    device = "cuda"
    
    # 创建输入张量
    input_tensor = torch.randn(4, 5, 6, dtype=dtype, device=device)
    dim = 1
    
    # 不使用 out 参数
    result_no_out = ntops.torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim)
    
    # 使用连续的 out 参数
    out_contiguous = torch.empty_like(result_no_out)
    result_with_out = ntops.torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim, out=out_contiguous)
    
    # 验证结果一致
    assert torch.allclose(result_no_out, result_with_out, rtol=1e-6, atol=1e-6)
    assert result_with_out is out_contiguous


