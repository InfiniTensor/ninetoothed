import pytest
import torch
import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments

@skip_if_cuda_not_available
@pytest.mark.parametrize("eps", (1e-5, 1e-3))
@pytest.mark.parametrize("affine", (True, False))
@pytest.mark.parametrize(*generate_arguments())
def test_batch_norm(
    shape, dtype, device, rtol, atol, affine, eps
):
    if len(shape) < 2:
        return
    
    input = torch.randn(shape, dtype=dtype, device=device)
    C = shape[1]
    
    if affine:
        weight = torch.randn(C, dtype=dtype, device=device)
        bias = torch.randn(C, dtype=dtype, device=device)
    else:
        weight = None
        bias = None

    # 调用 DSL 实现
    ninetoothed_output = ntops.torch.batch_norm(
        input, weight=weight, bias=bias, eps=eps, training=True
    )
    
    # 调用 PyTorch 参考
    # 必须指定 training=True 以强制从当前 batch 计算统计量
    reference_output = torch.nn.functional.batch_norm(
        input, 
        running_mean=None, 
        running_var=None, 
        weight=weight, 
        bias=bias, 
        training=True, 
        eps=eps
    )

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)