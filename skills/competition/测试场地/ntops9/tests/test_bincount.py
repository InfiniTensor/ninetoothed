import pytest
import torch
import ntops
from tests.skippers import skip_if_cuda_not_available

def generate_bincount_args():
    # 参数组合: (size, has_weights, minlength)
    cases = [
        (100, False, 0),
        (100, True, 0),
        (100, True, 50),
        (1000, False, 2000), # minlength > max_val
        (50, True, 10),      # minlength < max_val
        (0, False, 5),       # Empty input
    ]
    return "size, has_weights, minlength", cases

@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_bincount_args())
def test_bincount(size, has_weights, minlength):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 生成随机非负整数输入
    if size > 0:
        max_val = size  # 保证值有一定的分布
        input_tensor = torch.randint(0, max_val, (size,), device=device, dtype=torch.int32)
    else:
        input_tensor = torch.tensor([], device=device, dtype=torch.int32)
    
    weights = None
    if has_weights and size > 0:
        weights = torch.randn(size, device=device, dtype=torch.float32)
    elif has_weights and size == 0:
        weights = torch.tensor([], device=device, dtype=torch.float32)
    
    # 运行参考实现 (PyTorch)
    ref_out = torch.bincount(input_tensor, weights=weights, minlength=minlength)
    
    # 运行 ntops 实现
    ntops_out = ntops.torch.bincount(input_tensor, weights=weights, minlength=minlength)
    
    # 比较结果
    # 注意: 浮点数比较需要一定容差
    if ntops_out.is_floating_point():
        assert torch.allclose(ntops_out, ref_out, atol=1e-4)
    else:
        assert torch.equal(ntops_out, ref_out)