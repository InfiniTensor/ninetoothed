"""
自测任务 1：Element-wise 算子 — leaky_relu

算子类型：逐元素 + 标量参数
验证内容：float32/float16 精度、自定义 slope、边界情况
Benchmark：包含性能对比

输入任务说明：
    实现 leaky_relu 算子：f(x) = x if x >= 0, else negative_slope * x
    需要支持 negative_slope 参数（默认 0.01）

AI 智能体执行记录摘要：
    - 第 1 次：使用闭包传递 negative_slope → NameError
    - 第 2 次：使用 Tensor(0, dtype=float64) → IncompatibleTypeError
    - 第 3 次：改用 Tensor(0, constexpr=True, value=...) → 成功
    - 精度验证：float32/float16 全部通过
    - 性能：0.92x PyTorch（4096x4096）

Correctness 测试命令：
    cd $PROJECT_ROOT/ntops
    python -m pytest tests/test_leaky_relu.py -v
"""
import pytest
import torch
import ntops


DTYPE_TOLERANCES = [
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-3, 1e-3),
]


def cpu_leaky_relu(x, negative_slope=0.01):
    return torch.where(x >= 0, x, negative_slope * x)


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_leaky_relu_basic(dtype, rtol, atol):
    input = torch.randn(1024, device="cuda", dtype=dtype)
    result = ntops.torch.leaky_relu(input)
    reference = cpu_leaky_relu(input)

    assert torch.allclose(result, reference, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_leaky_relu_large(dtype, rtol, atol):
    input = torch.randn(4096, 1024, device="cuda", dtype=dtype)
    result = ntops.torch.leaky_relu(input)
    reference = cpu_leaky_relu(input)

    assert torch.allclose(result, reference, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_leaky_relu_custom_slope():
    input = torch.randn(512, 512, device="cuda", dtype=torch.float32)

    for slope in [0.01, 0.1, 0.5, 1.0, 2.0]:
        result = ntops.torch.leaky_relu(input, negative_slope=slope)
        reference = cpu_leaky_relu(input, negative_slope=slope)
        assert torch.allclose(result, reference, rtol=1e-5, atol=1e-5)


def test_leaky_relu_edge_cases():
    # 全零
    input = torch.zeros(256, device="cuda", dtype=torch.float32)
    result = ntops.torch.leaky_relu(input)
    assert torch.allclose(result, torch.zeros_like(input))

    # 全正
    input = torch.ones(256, device="cuda", dtype=torch.float32)
    result = ntops.torch.leaky_relu(input)
    assert torch.allclose(result, torch.ones_like(input))

    # 全负
    input = -torch.ones(256, device="cuda", dtype=torch.float32)
    result = ntops.torch.leaky_relu(input, negative_slope=0.5)
    reference = -0.5 * torch.ones_like(input)
    assert torch.allclose(result, reference, rtol=1e-5, atol=1e-5)


def test_leaky_relu_3d():
    input = torch.randn(8, 64, 128, device="cuda", dtype=torch.float32)
    result = ntops.torch.leaky_relu(input)
    reference = cpu_leaky_relu(input)
    assert torch.allclose(result, reference, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
