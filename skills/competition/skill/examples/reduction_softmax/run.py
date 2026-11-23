"""
reduction_softmax/run.py
Online softmax kernel: reduction_2d 模式的标准实现。

算法: Online softmax (numerically stable)
  1. Load BLOCK_SIZE 元素（整行）
  2. 计算 m = max(x), d = exp(x - m)
  3. 累加 sum_d = sum(d)
  4. output = d / sum_d

测试覆盖:
  - fp32/fp16
  - contiguous / non-contiguous
  - 各种行数/列数组合
  - 极端值场景
"""

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor


def make_softmax():
    """
    reduction_2d 模式: 输入 (M, N)，沿 N 维做归约。
    tile((1, BLOCK_SIZE)) 保留 M 维，N 维整行处理。
    BLOCK_SIZE 在 kernel 调用时传入（通常取 input.shape[-1]）。
    """
    BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

    def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
        return (
            input.tile((1, BLOCK_SIZE)),
            output.tile((1, BLOCK_SIZE)),
        )

    def application(input, output):
        x = input
        m = ntl.max(x, dim=-1, keepdim=True)
        d = ntl.exp(x - m)  # numerical stable
        sum_d = ntl.sum(d, dim=-1, keepdim=True)
        output = d / sum_d  # noqa: F841

    return ninetoothed.make(
        arrangement,
        application,
        tensors=(Tensor(2, other=float("-inf")), Tensor(2)),
    )


def make_softmax_no_autotune(BLOCK_SIZE=1024):
    """固定 BLOCK_SIZE 的版本。传入的 BLOCK_SIZE 在 kernel 调用时以 BS=... 传入。"""
    BS = Symbol("BS", constexpr=True)

    def arrangement(input, output, BS=BS):
        return (
            input.tile((1, BS)),
            output.tile((1, BS)),
        )

    def application(input, output):
        x = input
        m = ntl.max(x, dim=-1, keepdim=True)
        d = ntl.exp(x - m)
        sum_d = ntl.sum(d, dim=-1, keepdim=True)
        output = d / sum_d  # noqa: F841

    return ninetoothed.make(
        arrangement,
        application,
        tensors=(Tensor(2, other=float("-inf")), Tensor(2)),
    )


def demo():
    print("=" * 50)
    print("Softmax — Reduction 2D 示例")
    print("=" * 50)

    kernel = make_softmax()
    print("✅ Kernel created")

    # 测试 1: 基本 2D softmax
    x = torch.randn(4, 1024, device="cuda")
    out = torch.empty_like(x)
    kernel(x, out, BLOCK_SIZE=1024)
    expected = torch.softmax(x, dim=-1)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), "Basic softmax failed"
    print("✅ Basic softmax (4, 1024) — PASS")

    # 测试 2: 多行
    x = torch.randn(128, 1024, device="cuda")
    out = torch.empty_like(x)
    kernel(x, out, BLOCK_SIZE=1024)
    expected = torch.softmax(x, dim=-1)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), "Multi-row softmax failed"
    print("✅ Multi-row softmax (128, 1024) — PASS")

    # 测试 3: 列数不能被整除（需要 mask 处理）
    x_small = torch.randn(4, 999, device="cuda")
    kernel_small = make_softmax_no_autotune()
    out = torch.empty_like(x_small)
    kernel_small(x_small, out, BS=2048)  # 大于 shape[-1] 的 BLOCK_SIZE 测试 mask
    expected = torch.softmax(x_small, dim=-1)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), "Uneven col softmax failed"
    print("✅ Uneven cols (4, 999) with mask — PASS")

    # 测试 4: 单行
    x = torch.randn(1, 1024, device="cuda")
    out = torch.empty_like(x)
    kernel(x, out, BLOCK_SIZE=1024)
    expected = torch.softmax(x, dim=-1)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), "Single row softmax failed"
    print("✅ Single row (1, 1024) — PASS")

    # 测试 5: FP16
    x_fp16 = torch.randn(4, 1024, device="cuda", dtype=torch.float16)
    out = torch.empty_like(x_fp16)
    kernel(x_fp16, out, BLOCK_SIZE=1024)
    expected = torch.softmax(x_fp16, dim=-1)
    assert torch.allclose(out, expected, atol=1e-3, rtol=1e-3), "FP16 softmax failed"
    print("✅ FP16 (4, 1024) — PASS")

    # 测试 6: 极端值（大正值，测试数值稳定性）
    x_big = torch.full((4, 1024), 1000.0, device="cuda")
    out = torch.empty_like(x_big)
    kernel(x_big, out, BLOCK_SIZE=1024)
    expected = torch.softmax(x_big, dim=-1)
    assert torch.allclose(out, expected, atol=1e-3, rtol=1e-3), "Extreme values softmax failed"
    print("✅ Extreme values (4, 1024) fill=1000 — PASS")

    # 测试 7: Non-contiguous (strided)
    x_nc = torch.randn(8, 2048, device="cuda")[:, ::2]  # shape (8, 1024), non-contiguous
    kernel_nc = make_softmax()
    out = torch.empty(8, 1024, device="cuda")
    kernel_nc(x_nc, out, BLOCK_SIZE=1024)
    expected = torch.softmax(x_nc, dim=-1)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), "Non-contiguous softmax failed"
    print("✅ Non-contiguous (8, 2048)[:, ::2] — PASS")

    # 测试 8: 质数列数
    x_prime = torch.randn(3, 17, device="cuda")
    out = torch.empty_like(x_prime)
    kernel_prime = make_softmax_no_autotune()
    kernel_prime(x_prime, out, BS=32)
    expected = torch.softmax(x_prime, dim=-1)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), "Prime cols softmax failed"
    print("✅ Prime cols (3, 17) — PASS")

    print()
    print("🎉 所有测试通过！")


if __name__ == "__main__":
    demo()
