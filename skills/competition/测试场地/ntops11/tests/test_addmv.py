import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
# 假设你有一个类似的参数生成器，或者我们需要稍微修改以适应 addmv (n=1)
from tests.test_mm import generate_arguments 
from tests.utils import gauss

@skip_if_cuda_not_available
# 我们可以复用 generate_arguments，但在 addmv 中，n 维度通常不再是矩阵列数，而是 1 (或者不需要)
# 这里假设 generate_arguments 返回 (m, n, k, ...)，我们在 test 中忽略 n 或者将其视为 batch
@pytest.mark.parametrize(*generate_arguments()) 
def test_addmv(m, n, k, dtype, device, rtol, atol):
    # 构造数据
    # input: (m,) - 偏置向量
    input = torch.randn((m,), dtype=dtype, device=device)
    # mat: (m, k) - 权重矩阵
    mat = torch.randn((m, k), dtype=dtype, device=device)
    # vec: (k,) - 输入向量
    vec = torch.randn((k,), dtype=dtype, device=device)
    
    beta = gauss()
    alpha = gauss()

    # 计算 ninetoothed 结果
    ninetoothed_output = ntops.torch.addmv(input, mat, vec, beta=beta, alpha=alpha)
    
    # 计算 PyTorch 参考结果
    reference_output = torch.addmv(input, mat, vec, beta=beta, alpha=alpha)

    # 验证结果
    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)