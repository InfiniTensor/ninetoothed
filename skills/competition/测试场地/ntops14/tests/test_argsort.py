import random
import pytest
import torch
import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import _random_shape

# 生成测试参数
def generate_argsort_args():
    args = []
    for dtype in (torch.float32, torch.float16):
        device = "cuda"
        rtol, atol = (1e-3, 1e-3) if dtype == torch.float32 else (1e-2, 1e-2)

        for ndim in range(1, 4):
            for _ in range(5):
                shape = _random_shape(ndim)
                dim = random.randint(0, ndim - 1)
                
                # 限制 dim_size 以免测试跑太久 (因为选择排序是 O(N^2))
                # 对于 DSL Demo 这是一个合理的限制
                if shape[dim] > 128:
                    shape_list = list(shape)
                    shape_list[dim] = random.randint(10, 128)
                    shape = tuple(shape_list)

                args.append((shape, dim, dtype, device, rtol, atol))
    return "shape, dim, dtype, device, rtol, atol", args

@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_argsort_args())
@pytest.mark.parametrize("descending", [True, False])
def test_argsort(shape, dim, dtype, device, rtol, atol, descending):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)

    # 调用 ntops 实现
    # 注意：PyTorch 的 argsort 默认是 ascending (descending=False)
    ntops_indices = ntops.torch.argsort(input_tensor, dim=dim, descending=descending)
    
    # 调用 PyTorch 参考实现
    ref_indices = torch.argsort(input_tensor, dim=dim, descending=descending)

    # 验证索引对应的值是否是有序的
    # 直接比较 indices 可能会失败，因为对于相同的值，排序是不稳定的 (unstable)
    # 所以我们 gather 出值来比较值是否一致且有序
    
    ntops_gathered = torch.gather(input_tensor, dim, ntops_indices)
    ref_gathered = torch.gather(input_tensor, dim, ref_indices)

    assert torch.allclose(ntops_gathered, ref_gathered, rtol=rtol, atol=atol)
    
    # 额外验证：确保输出确实是排序过的
    if input_tensor.numel() > 0:
        diffs = ntops_gathered.diff(dim=dim)
        if descending:
            assert (diffs <= atol).all(), "Result is not sorted descending"
        else:
            assert (diffs >= -atol).all(), "Result is not sorted ascending"