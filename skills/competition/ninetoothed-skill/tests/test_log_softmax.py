"""
自测任务 2：Reduction 算子 — log_softmax

算子类型：归约 + 数值稳定性
验证内容：float32/float16 精度、多维度、数值稳定性、边界情况
Benchmark：包含性能对比

输入任务说明：
    实现 log_softmax 算子：log(softmax(x, dim))
    需要数值稳定实现（online max + log-sum-exp）

AI 智能体执行记录摘要：
    - 一次成功，基于 softmax 模式扩展
    - 使用 online softmax 算法保证数值稳定性
    - float16 中间计算提升到 float32
    - 精度验证：float32/float16 全部通过
    - 性能：0.81x PyTorch（4096x1024）

Correctness 测试命令：
    cd $PROJECT_ROOT/ntops
    python -m pytest tests/test_log_softmax.py -v
"""
import pytest
import torch
import ntops


DTYPE_TOLERANCES = [
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-3, 1e-3),
]


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_log_softmax_basic(dtype, rtol, atol):
    input = torch.randn(128, 1024, device="cuda", dtype=dtype)
    result = ntops.torch.log_softmax(input, dim=-1)
    reference = torch.log_softmax(input, dim=-1)

    assert torch.allclose(result, reference, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_log_softmax_large(dtype, rtol, atol):
    input = torch.randn(4096, 1024, device="cuda", dtype=dtype)
    result = ntops.torch.log_softmax(input, dim=-1)
    reference = torch.log_softmax(input, dim=-1)

    assert torch.allclose(result, reference, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_log_softmax_dim0():
    input = torch.randn(512, 512, device="cuda", dtype=torch.float32)
    result = ntops.torch.log_softmax(input, dim=0)
    reference = torch.log_softmax(input, dim=0)
    assert torch.allclose(result, reference, rtol=1e-5, atol=1e-5)


def test_log_softmax_3d():
    input = torch.randn(8, 64, 128, device="cuda", dtype=torch.float32)
    result = ntops.torch.log_softmax(input, dim=-1)
    reference = torch.log_softmax(input, dim=-1)
    assert torch.allclose(result, reference, rtol=1e-5, atol=1e-5)


def test_log_softmax_numerical_stability():
    # 极端值测试
    input = torch.tensor([
        [-10000.0, 0.0, 10000.0],
        [-10000.0, -10000.0, -10000.0],
    ], device="cuda", dtype=torch.float32)
    result = ntops.torch.log_softmax(input, dim=-1)
    reference = torch.log_softmax(input, dim=-1)

    assert torch.allclose(result, reference, rtol=1e-4, atol=1e-4)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_log_softmax_edge_cases():
    # 小尺寸
    input = torch.randn(1, 4, device="cuda", dtype=torch.float32)
    result = ntops.torch.log_softmax(input, dim=-1)
    reference = torch.log_softmax(input, dim=-1)
    assert torch.allclose(result, reference, rtol=1e-5, atol=1e-5)

    # size 不整除 block_size
    input = torch.randn(100, 333, device="cuda", dtype=torch.float32)
    result = ntops.torch.log_softmax(input, dim=-1)
    reference = torch.log_softmax(input, dim=-1)
    assert torch.allclose(result, reference, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
