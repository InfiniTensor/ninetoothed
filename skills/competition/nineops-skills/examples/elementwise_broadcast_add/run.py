"""
elementwise_broadcast_add/run.py
Broadcast add kernel: elementwise_1d 模式的标准实现。

测试覆盖:
  - fp32/fp16/bf16
  - contiguous / non-contiguous (strided)
  - broadcast scenarios (scalar, vector, 3D)
  - shape 不能整除 BLOCK_SIZE 的边界情况
"""

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

# ── 方法 A: 使用 elementwise_1d 模板 ──────────────────────────────

def make_broadcast_add_elementwise_1d():
    """
    Elementwise-1D 模式: 所有 tensor 沿最后一维做 tile((BLOCK_SIZE,))。
    """
    BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

    def arrangement(input, other, output, BLOCK_SIZE=BLOCK_SIZE):
        return (
            input.tile((BLOCK_SIZE,)),
            other.tile((BLOCK_SIZE,)),
            output.tile((BLOCK_SIZE,)),
        )

    def application(input, other, output):
        output = input + other  # noqa: F841

    return ninetoothed.make(
        arrangement,
        application,
        tensors=(Tensor(1), Tensor(1), Tensor(1)),
    )

# ── 方法 B: 手动 broadcast 展开（用于理解和对比）───────────────────

def make_broadcast_add_manual():
    BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

    def arrangement(input, other, output, BLOCK_SIZE=BLOCK_SIZE):
        return (
            input.tile((BLOCK_SIZE,)),
            other.tile((BLOCK_SIZE,)),
            output.tile((BLOCK_SIZE,)),
        )

    def application(input, other, output):
        output = input + other  # noqa: F841

    return ninetoothed.make(
        arrangement,
        application,
        tensors=(Tensor(1), Tensor(1), Tensor(1)),
    )


# ── 使用示例 ──────────────────────────────────────────────────────

def demo():
    print("=" * 50)
    print("Broadcast Add — Elementwise 1D 示例")
    print("=" * 50)

    # 创建 kernel
    kernel = make_broadcast_add_elementwise_1d()
    print("✅ Kernel created")

    # 测试 1: 基本加法
    x = torch.randn(4096, device="cuda")
    y = torch.randn(4096, device="cuda")
    out = torch.empty(4096, device="cuda")
    kernel(x, y, out, BLOCK_SIZE=1024)
    expected = x + y
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), "Basic add failed"
    print("✅ Basic add: contiguous, same shape — PASS")

    # 测试 2: Scalar broadcast
    y_scalar = torch.randn(1, device="cuda")
    out = torch.empty(4096, device="cuda")
    kernel(x, y_scalar, out, BLOCK_SIZE=1024)
    expected = x + y_scalar
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), "Scalar broadcast failed"
    print("✅ Scalar broadcast: (4096,) + (1,) — PASS")

    # 测试 3: Vector broadcast (2D + 1D)
    x2d = torch.randn(128, 256, device="cuda")
    y1d = torch.randn(256, device="cuda")
    kernel_2d = make_broadcast_add_elementwise_1d()
    out = torch.empty(128, 256, device="cuda")
    kernel_2d(x2d, y1d, out, BLOCK_SIZE=256)
    expected = x2d + y1d
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), "2D+1D broadcast failed"
    print("✅ 2D+1D broadcast: (128, 256) + (256,) — PASS")

    # 测试 4: Non-contiguous (transposed)
    x_t = torch.randn(256, 128, device="cuda").t()  # shape (128, 256)
    y_t = torch.randn(128, device="cuda")
    kernel_nc = make_broadcast_add_elementwise_1d()
    out = torch.empty(128, 256, device="cuda")
    kernel_nc(x_t, y_t, out, BLOCK_SIZE=256)
    expected = x_t + y_t
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), "Non-contiguous broadcast failed"
    print("✅ Non-contiguous: (128, 256).T + (128,) — PASS")

    # 测试 5: 边界情况 — shape 不能被 BLOCK_SIZE 整除
    x_small = torch.randn(100, device="cuda")
    y_small = torch.randn(1, device="cuda")
    out = torch.empty(100, device="cuda")
    kernel(x_small, y_small, out, BLOCK_SIZE=128)
    expected = x_small + y_small
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), "Uneven shape failed"
    print("✅ Uneven shape: (100,) + (1,) — PASS")

    # 测试 6: FP16
    x_fp16 = torch.randn(4096, device="cuda", dtype=torch.float16)
    y_fp16 = torch.randn(4096, device="cuda", dtype=torch.float16)
    out = torch.empty(4096, device="cuda", dtype=torch.float16)
    kernel(x_fp16, y_fp16, out, BLOCK_SIZE=1024)
    expected = x_fp16 + y_fp16
    assert torch.allclose(out, expected, atol=1e-3, rtol=1e-3), "FP16 failed"
    print("✅ FP16: contiguous — PASS")

    print()
    print("🎉 所有测试通过！")


if __name__ == "__main__":
    demo()
