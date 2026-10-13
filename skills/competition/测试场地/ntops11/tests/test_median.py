import random
import pytest
import torch
import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import _random_shape

def generate_median_args():
    args = []
    # Median 也是一种选择排序，复杂度较高，测试时控制 dim_size
    for dtype in (torch.float32, torch.float16):
        device = "cuda"
        rtol, atol = (1e-3, 1e-3) if dtype == torch.float32 else (1e-2, 1e-2)

        for ndim in range(1, 4):
            for _ in range(5):
                shape = _random_shape(ndim)
                dim = random.randint(0, ndim - 1)
                
                # 限制 dim_size，原因同 argsort
                if shape[dim] > 128:
                    shape_list = list(shape)
                    shape_list[dim] = random.randint(10, 128)
                    shape = tuple(shape_list)

                args.append((shape, dim, dtype, device, rtol, atol))
    return "shape, dim, dtype, device, rtol, atol", args

@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_median_args())
def test_median(shape, dim, dtype, device, rtol, atol):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)

    # 1. 调用 ntops 实现
    ntops_vals, ntops_idxs = ntops.torch.median(input_tensor, dim=dim, keepdim=False)
    
    # 2. 调用 PyTorch 参考实现
    ref_vals, ref_idxs = torch.median(input_tensor, dim=dim, keepdim=False)

    # 3. 验证值 (Values)
    # 浮点数比较需要容差
    assert torch.allclose(ntops_vals, ref_vals, rtol=rtol, atol=atol), "Values mismatch"

    # 4. 验证索引 (Indices)
    # 注意：如果有重复值，median 的 index 可能不唯一。
    # 更稳健的测试方法是：用 ntops 返回的 index 去 input 取值，看取出来的值是否等于 ref_val
    gathered_vals = torch.gather(input_tensor, dim, ntops_idxs.unsqueeze(dim)).squeeze(dim)
    assert torch.allclose(gathered_vals, ref_vals, rtol=rtol, atol=atol), "Gathered values from indices mismatch"
    
    # 如果数据没有重复值，可以直接比较 index (可选)
    # assert torch.equal(ntops_idxs, ref_idxs)