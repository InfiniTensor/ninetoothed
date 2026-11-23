"""
non_contiguous_stride_case/run.py
测试 ninetoothed 在 non-contiguous tensor 上的 stride 处理。

使用 add kernel（最简单的 elementwise）来聚焦 stride 行为测试。
"""

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor


def make_add_kernel():
    """简单的 elementwise add kernel，用于测试 stride。"""
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


def make_2d_add_kernel():
    """2D version, tile per row."""
    BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

    def arrangement(input, other, output, BLOCK_SIZE=BLOCK_SIZE):
        return (
            input.tile((1, BLOCK_SIZE)),
            other.tile((1, BLOCK_SIZE)),
            output.tile((1, BLOCK_SIZE)),
        )

    def application(input, other, output):
        output = input + other  # noqa: F841

    return ninetoothed.make(
        arrangement,
        application,
        tensors=(Tensor(2), Tensor(2), Tensor(2)),
    )


def test_stride_case(name, x, y=None, kernel=None, BLOCK_SIZE=1024):
    """通用 stride 测试函数。

    因为现在 kernel 需要 output 参数，由本函数自动分配。
    """
    if y is None:
        y = torch.randn(x.shape[-1], device=x.device)
    if kernel is None:
        kernel = make_add_kernel() if x.ndim == 1 else make_2d_add_kernel()

    out = torch.empty_like(x)
    try:
        kernel(x, y, out, BLOCK_SIZE=BLOCK_SIZE)
        expected = x + y
        ok = torch.allclose(out, expected, atol=1e-5, rtol=1e-5)
        status = "✅" if ok else "❌"
        extra = f" | strides: {x.stride()}" if not ok else ""
        print(f"  {status} {name}{extra}")
        return ok
    except Exception as e:
        print(f"  ❌ {name} — CRASH: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo():
    print("=" * 60)
    print("Stride / Non-Contiguous 测试")
    print("=" * 60)

    device = "cuda"
    results = []

    # ── 1D tests ──
    print("\n--- 1D ---")
    kernel_1d = make_add_kernel()

    results.append(test_stride_case("contiguous 1D",
        torch.randn(4096, device=device), kernel=kernel_1d))
    results.append(test_stride_case("sliced 1D [::2]",
        torch.randn(8192, device=device)[::2], kernel=kernel_1d))
    results.append(test_stride_case("sliced 1D [::3]",
        torch.randn(12288, device=device)[::3], kernel=kernel_1d))
    results.append(test_stride_case("small 1D (100)",
        torch.randn(100, device=device), kernel=kernel_1d))

    # ── 2D tests ──
    print("\n--- 2D ---")
    kernel_2d = make_2d_add_kernel()

    results.append(test_stride_case("contiguous (128, 256)",
        torch.randn(128, 256, device=device), kernel=kernel_2d))
    results.append(test_stride_case("transposed (256, 128).t()",
        torch.randn(256, 128, device=device).t(), kernel=kernel_2d))
    results.append(test_stride_case("sliced rows [::2]",
        torch.randn(256, 256, device=device)[::2, :], kernel=kernel_2d))
    results.append(test_stride_case("sliced cols [::2]",
        torch.randn(128, 512, device=device)[:, ::2], kernel=kernel_2d))
    results.append(test_stride_case("both sliced [::2, ::3]",
        torch.randn(256, 768, device=device)[::2, ::3], kernel=kernel_2d))

    # view
    x_view = torch.randn(128, 256, device=device).view(128, 256)
    results.append(test_stride_case("view (128, 256)",
        x_view, kernel=kernel_2d))

    # expanded (broadcast in add)
    x_base = torch.randn(1, 256, device=device)
    y_base = torch.randn(128, 1, device=device)

    # For expanded tensor, we create one then test
    x_exp = x_base.expand(128, 256)
    y_exp = y_base.expand(128, 256)
    kernel_exp = make_2d_add_kernel()
    out = torch.empty_like(x_exp)
    kernel_exp(x_exp, y_exp, out, BLOCK_SIZE=256)
    expected = x_exp + y_exp
    ok = torch.allclose(out, expected, atol=1e-5, rtol=1e-5)
    status = "✅" if ok else "❌"
    print(f"  {status} expanded broadcast (1,256)+(128,1) — {'PASS' if ok else 'FAIL'}")
    results.append(ok)

    # permuted
    x_perm = torch.randn(128, 256, device=device).permute(1, 0)
    results.append(test_stride_case("permuted (1,0)",
        x_perm, kernel=kernel_2d))

    # as_strided
    x_base2 = torch.randn(256, 256, device=device)
    x_as = torch.as_strided(x_base2, (128, 128), (512, 2))  # custom stride
    results.append(test_stride_case("as_strided (128,128) stride(512,2)",
        x_as, kernel=kernel_2d))

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"结果: {sum(results)}/{len(results)} 通过")

    if all(results):
        print("✅ 所有 stride 变体均正确处理！")
    else:
        print(f"❌ {sum(1 for r in results if not r)} 个失败")


if __name__ == "__main__":
    demo()
