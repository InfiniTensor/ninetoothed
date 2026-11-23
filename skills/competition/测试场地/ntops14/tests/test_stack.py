import random
import pytest
import torch
import ntops
from tests.skippers import skip_if_cuda_not_available

@skip_if_cuda_not_available
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_stack(dim):
    device = "cuda"
    dtype = torch.float32
    
    # 构造测试参数
    num_tensors = random.randint(2, 5)
    shape = (random.randint(16, 64), random.randint(16, 64), random.randint(16, 64))
    
    # 确保 dim 不越界 (ndim + 1 因为 stack 会增加维度)
    if dim > len(shape):
        return

    # 创建输入列表
    tensors = [
        torch.randn(shape, device=device, dtype=dtype) 
        for _ in range(num_tensors)
    ]

    # Reference implementation
    reference_output = torch.stack(tensors, dim=dim)

    # Ntops implementation
    ntops_output = ntops.torch.stack(tensors, dim=dim)

    # 验证一致性
    assert torch.allclose(ntops_output, reference_output, atol=1e-6, rtol=1e-6)
    
    # 验证内存是否连续（Stack 的输出通常是连续的）
    assert ntops_output.is_contiguous()