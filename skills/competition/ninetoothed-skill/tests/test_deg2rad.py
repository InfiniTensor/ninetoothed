"""
自测任务：Element-wise 算子 — deg2rad

算子类型：逐元素（Pattern 1）
CPU 参考：deg2rad(x) = x * (pi / 180.0)
验证内容：float32/float16 精度、边界情况、大尺寸

测试命令：
    cd cleanEnv
    KMP_DUPLICATE_LIB_OK=TRUE python -m pytest ninetoothed-skill/tests/test_deg2rad.py -v
"""
import math

import pytest
import torch
import ntops


DTYPE_TOLERANCES = [
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-3, 1e-3),
]


def cpu_deg2rad(x):
    return x * (math.pi / 180.0)


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_deg2rad_basic(dtype, rtol, atol):
    input = torch.randn(1024, device="cuda", dtype=dtype)
    result = ntops.torch.deg2rad(input)
    reference = cpu_deg2rad(input)

    assert torch.allclose(result, reference, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_deg2rad_large(dtype, rtol, atol):
    input = torch.randn(4096, 4096, device="cuda", dtype=dtype)
    result = ntops.torch.deg2rad(input)
    reference = cpu_deg2rad(input)

    assert torch.allclose(result, reference, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_deg2rad_exact_values():
    """使用精确值验证：0°, 90°, 180°, 360°"""
    input = torch.tensor([0.0, 90.0, 180.0, 360.0], device="cuda", dtype=torch.float32)
    result = ntops.torch.deg2rad(input)
    expected = torch.tensor([0.0, math.pi/2, math.pi, 2*math.pi], device="cuda", dtype=torch.float32)
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_deg2rad_edge_cases():
    # 全零
    input = torch.zeros(256, device="cuda", dtype=torch.float32)
    result = ntops.torch.deg2rad(input)
    assert torch.allclose(result, torch.zeros_like(input))

    # 负值
    input = torch.tensor([-180.0, -90.0, -45.0], device="cuda", dtype=torch.float32)
    result = ntops.torch.deg2rad(input)
    expected = torch.tensor([-math.pi, -math.pi/2, -math.pi/4], device="cuda", dtype=torch.float32)
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_deg2rad_3d():
    input = torch.randn(8, 64, 128, device="cuda", dtype=torch.float32)
    result = ntops.torch.deg2rad(input)
    reference = cpu_deg2rad(input)
    assert torch.allclose(result, reference, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
