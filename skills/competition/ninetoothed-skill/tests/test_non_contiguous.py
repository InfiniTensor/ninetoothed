"""
自测任务 3：非连续输入 / 步长 / 偏移量场景

验证内容：
    - 转置张量（非连续 stride）
    - 步幅切片（strided slice）
    - 多种算子类型（element-wise unary, binary, reduction）
    - 确认 NineToothed 自动处理 stride 信息

AI 智能体执行记录摘要：
    - 测试覆盖 relu, silu, add, leaky_relu（element-wise）和 softmax（reduction）
    - 所有算子均正确处理非连续输入，无需 .contiguous()
    - NineToothed 自动处理 stride 信息，torch 层无需特殊处理

Correctness 测试命令：
    cd $PROJECT_ROOT/ntops
    python -m pytest tests/test_non_contiguous.py -v
"""
import pytest
import torch
import ntops


DTYPE_TOLERANCES = [
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-3, 1e-3),
]


# --- Element-wise unary: relu ---

@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_relu_transposed(dtype, rtol, atol):
    input = torch.randn(256, 512, device="cuda", dtype=dtype)
    input_t = input.t()  # 非连续

    assert not input_t.is_contiguous()

    result = ntops.torch.relu(input_t)
    reference = torch.relu(input_t)
    assert torch.allclose(result, reference, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_relu_strided_slice(dtype, rtol, atol):
    input = torch.randn(512, 512, device="cuda", dtype=dtype)
    sliced = input[::2, ::3]  # 非连续步幅切片

    assert not sliced.is_contiguous()

    result = ntops.torch.relu(sliced)
    reference = torch.relu(sliced)
    assert torch.allclose(result, reference, rtol=rtol, atol=atol)


# --- Element-wise unary: silu ---

@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_silu_transposed(dtype, rtol, atol):
    input = torch.randn(256, 512, device="cuda", dtype=dtype)
    input_t = input.t()

    result = ntops.torch.silu(input_t)
    reference = torch.nn.functional.silu(input_t)
    assert torch.allclose(result, reference, rtol=rtol, atol=atol)


# --- Element-wise binary: add ---

@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_add_transposed(dtype, rtol, atol):
    a = torch.randn(256, 512, device="cuda", dtype=dtype)
    b = torch.randn(256, 512, device="cuda", dtype=dtype)
    a_t = a.t()
    b_t = b.t()

    result = ntops.torch.add(a_t, b_t)
    reference = a_t + b_t
    assert torch.allclose(result, reference, rtol=rtol, atol=atol)


# --- Element-wise with scalar: leaky_relu ---

@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_leaky_relu_transposed(dtype, rtol, atol):
    input = torch.randn(256, 512, device="cuda", dtype=dtype)
    input_t = input.t()

    result = ntops.torch.leaky_relu(input_t, negative_slope=0.1)
    reference = torch.nn.functional.leaky_relu(input_t, negative_slope=0.1)
    assert torch.allclose(result, reference, rtol=rtol, atol=atol)


# --- Reduction: softmax ---

@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_softmax_transposed(dtype, rtol, atol):
    input = torch.randn(512, 256, device="cuda", dtype=dtype)
    input_t = input.t()

    result = ntops.torch.softmax(input_t, dim=-1)
    reference = torch.softmax(input_t, dim=-1)
    assert torch.allclose(result, reference, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
