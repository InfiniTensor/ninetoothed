#!/usr/bin/env python3
"""
九齿算子完整测试套件

在有 CUDA 的环境中运行此脚本来验证所有算子是否正确工作。

用法：
    python run_all_operator_tests.py
"""

import torch
import sys
import time

# 导入所有算子
from operators import (
    create_add_kernel,
    create_2d_add_kernel,
    create_relu_kernel,
    create_sigmoid_kernel,
    create_gelu_kernel,
    create_softmax_kernel,
    create_sum_kernel,
    create_strided_add_kernel,
    create_2d_strided_add_kernel,
    create_rms_norm_kernel,
)


def check_cuda():
    """检查 CUDA 是否可用"""
    if not torch.cuda.is_available():
        print("❌ 错误: CUDA 不可用！")
        print("请在有 CUDA 的环境中运行此脚本。")
        sys.exit(1)
    print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA 版本: {torch.version.cuda}")
    print()


def test_add_1d():
    """测试一维向量加法"""
    print("=" * 60)
    print("测试 1: 一维向量加法 (Add 1D)")
    print("=" * 60)

    kernel = create_add_kernel()
    shapes = [(100,), (512,), (1024,)]

    for shape in shapes:
        a = torch.randn(shape, dtype=torch.float16, device='cuda')
        b = torch.randn(shape, dtype=torch.float16, device='cuda')
        c = torch.empty_like(a)

        kernel(a, b, c)
        expected = a + b

        is_close = torch.allclose(c, expected, atol=1e-5, rtol=1e-3)
        status = "✓ 通过" if is_close else "✗ 失败"
        print(f"  Shape {str(shape):15s}: {status}")

        if not is_close:
            diff = torch.abs(c - expected)
            print(f"    最大差异: {diff.max().item():.6e}")

    # 2D 测试：沿最后一维 stride=2
    print("  2D 测试 (沿列方向步长):")
    kernel_2d = create_2d_strided_add_kernel()
    shapes_2d = [(32, 64), (64, 128), (128, 256)]

    for shape in shapes_2d:
        a = torch.randn(shape, dtype=torch.float16, device='cuda')
        b = torch.randn(shape, dtype=torch.float16, device='cuda')
        c = torch.zeros_like(a)

        kernel_2d(a, b, c, BLOCK_SIZE_ROW=shape[0], BLOCK_SIZE_COL=shape[1] // 2)
        # 手动计算: c[:, ::2] = a[:, ::2] + b[:, ::2], 其他位置为 0
        expected = torch.zeros_like(a)
        expected[:, ::2] = a[:, ::2] + b[:, ::2]

        is_close = torch.allclose(c, expected, atol=1e-5, rtol=1e-3)
        status = "✓ 通过" if is_close else "✗ 失败"
        print(f"  Shape {str(shape):15s} [:, ::2]: {status}")

        if not is_close:
            diff = torch.abs(c - expected)
            print(f"    最大差异: {diff.max().item():.6e}")

    print()
    return True


def test_add_2d():
    """测试二维矩阵加法"""
    print("=" * 60)
    print("测试 2: 二维矩阵加法 (Add 2D)")
    print("=" * 60)

    kernel = create_2d_add_kernel()
    shapes = [(100, 50), (512, 512), (1024, 1024)]

    for shape in shapes:
        a = torch.randn(shape, dtype=torch.float16, device='cuda')
        b = torch.randn(shape, dtype=torch.float16, device='cuda')
        c = torch.empty_like(a)

        kernel(a, b, c)
        expected = a + b

        is_close = torch.allclose(c, expected, atol=1e-5, rtol=1e-3)
        status = "✓ 通过" if is_close else "✗ 失败"
        print(f"  Shape {str(shape):15s}: {status}")

        if not is_close:
            diff = torch.abs(c - expected)
            print(f"    最大差异: {diff.max().item():.6e}")

    print()
    return True


def test_relu():
    """测试 ReLU 激活函数"""
    print("=" * 60)
    print("测试 3: ReLU 激活函数")
    print("=" * 60)

    kernel = create_relu_kernel()
    shapes = [(100,), (512,), (1024,)]

    for shape in shapes:
        x = torch.randn(shape, dtype=torch.float16, device='cuda')
        output = torch.empty_like(x)

        kernel(x, output)
        expected = torch.nn.functional.relu(x)

        is_close = torch.allclose(output, expected, atol=1e-5, rtol=1e-3)
        status = "✓ 通过" if is_close else "✗ 失败"
        print(f"  Shape {str(shape):15s}: {status}")

        if not is_close:
            diff = torch.abs(output - expected)
            print(f"    最大差异: {diff.max().item():.6e}")

    print()
    return True


def test_sigmoid():
    """测试 Sigmoid 激活函数"""
    print("=" * 60)
    print("测试 4: Sigmoid 激活函数")
    print("=" * 60)

    kernel = create_sigmoid_kernel()
    shapes = [(100,), (512,), (1024,)]

    for shape in shapes:
        x = torch.randn(shape, dtype=torch.float16, device='cuda')
        output = torch.empty_like(x)

        kernel(x, output)
        expected = torch.sigmoid(x)

        is_close = torch.allclose(output, expected, atol=1e-2, rtol=1e-2)
        status = "✓ 通过" if is_close else "✗ 失败"
        print(f"  Shape {str(shape):15s}: {status}")

        if not is_close:
            diff = torch.abs(output - expected)
            print(f"    最大差异: {diff.max().item():.6e}")

    print()
    return True


def test_gelu():
    """测试 GELU 激活函数"""
    print("=" * 60)
    print("测试 5: GELU 激活函数")
    print("=" * 60)

    kernel = create_gelu_kernel()
    shapes = [(100,), (512,), (1024,)]

    for shape in shapes:
        x = torch.randn(shape, dtype=torch.float16, device='cuda')
        output = torch.empty_like(x)

        kernel(x, output)
        expected = torch.nn.functional.gelu(x)

        is_close = torch.allclose(output, expected, atol=1e-2, rtol=1e-2)
        status = "✓ 通过" if is_close else "✗ 失败"
        print(f"  Shape {str(shape):15s}: {status}")

        if not is_close:
            diff = torch.abs(output - expected)
            print(f"    最大差异: {diff.max().item():.6e}")

    print()
    return True


def test_softmax():
    """测试 Softmax 归约函数"""
    print("=" * 60)
    print("测试 6: Softmax 归约函数")
    print("=" * 60)

    kernel = create_softmax_kernel()
    shapes = [(512, 256), (1024, 512)]

    for shape in shapes:
        x = torch.randn(shape, dtype=torch.float16, device='cuda')
        output = torch.empty_like(x)

        kernel(x, output, BLOCK_SIZE=x.shape[-1])
        expected = torch.softmax(x, dim=-1)

        is_close = torch.allclose(output, expected, atol=1e-2, rtol=1e-2)
        status = "✓ 通过" if is_close else "✗ 失败"
        print(f"  Shape {str(shape):15s}: {status}")

        if not is_close:
            diff = torch.abs(output - expected)
            print(f"    最大差异: {diff.max().item():.6e}")

        # 验证输出和为 1
        sum_output = output.sum(dim=-1)
        is_sum_one = torch.allclose(sum_output, torch.ones_like(sum_output), atol=1e-2, rtol=1e-2)
        sum_status = "✓ 和为1" if is_sum_one else "✗ 和不为1"
        print(f"  Shape {str(shape):15s}: {sum_status}")

    print()
    return True


def test_sum():
    """测试 Sum 归约函数"""
    print("=" * 60)
    print("测试 7: Sum 归约函数")
    print("=" * 60)

    kernel = create_sum_kernel()
    shapes = [(100,), (512,), (1024,)]

    for shape in shapes:
        x = torch.randn(shape, dtype=torch.float16, device='cuda')
        output = torch.empty(1, dtype=torch.float16, device='cuda')

        kernel(x, output, BLOCK_SIZE=x.numel())
        expected = x.sum()

        is_close = torch.allclose(output, expected, atol=5e-2, rtol=1e-2)
        status = "✓ 通过" if is_close else "✗ 失败"
        print(f"  Shape {str(shape):15s}: {status}")

        if not is_close:
            diff = torch.abs(output - expected)
            print(f"    最大差异: {diff.max().item():.6e}")

    print()
    return True


def test_strided_add():
    """测试 Strided Add（非连续/步长）"""
    print("=" * 60)
    print("测试 8: Strided Add（非连续/步长）")
    print("=" * 60)

    kernel = create_strided_add_kernel()
    # stride=2: 只处理偶数位置元素
    shapes = [(100,), (512,), (1024,)]

    for shape in shapes:
        a = torch.randn(shape, dtype=torch.float16, device='cuda')
        b = torch.randn(shape, dtype=torch.float16, device='cuda')
        c = torch.zeros_like(a)

        kernel(a, b, c, BLOCK_SIZE=shape[0])
        # 手动计算: c[::2] = a[::2] + b[::2], 其他位置为 0
        expected = torch.zeros_like(a)
        expected[::2] = a[::2] + b[::2]

        is_close = torch.allclose(c, expected, atol=1e-5, rtol=1e-3)
        status = "✓ 通过" if is_close else "✗ 失败"
        print(f"  Shape {str(shape):15s} stride=2: {status}")

        if not is_close:
            diff = torch.abs(c - expected)
            print(f"    最大差异: {diff.max().item():.6e}")

    print()
    return True


def test_rms_norm():
    """测试 RMS Normalization（融合归约）"""
    print("=" * 60)
    print("测试 9: RMS Normalization（融合归约）")
    print("=" * 60)

    kernel = create_rms_norm_kernel()
    shapes = [(128, 256), (512, 512), (1024, 512)]

    for shape in shapes:
        x = torch.randn(shape, dtype=torch.float16, device='cuda')
        w = torch.randn(shape[-1], dtype=torch.float16, device='cuda')
        eps = 1e-5
        y = torch.empty_like(x)

        # 按官方约定：kernel 要求 w 为 2D，需先 expand
        kernel(x, w.expand_as(x), eps, y, BLOCK_SIZE=x.shape[-1])

        # 手动计算 RMS Norm: y = x * rsqrt(mean(x^2) + eps) * w
        x_fp32 = x.float()
        expected = x_fp32 * torch.rsqrt(
            (x_fp32 * x_fp32).mean(dim=-1, keepdim=True) + eps
        ) * w

        is_close = torch.allclose(y, expected.half(), atol=1e-2, rtol=1e-2)
        status = "✓ 通过" if is_close else "✗ 失败"
        print(f"  Shape {str(shape):15s}: {status}")

        if not is_close:
            diff = torch.abs(y - expected.half())
            print(f"    最大差异: {diff.max().item():.6e}")

    print()
    return True


def main():
    """主函数"""
    print()
    print("=" * 60)
    print("九齿算子完整测试套件")
    print("=" * 60)
    print()

    # 检查 CUDA
    check_cuda()

    # 运行所有测试
    tests = [
        ("一维向量加法", test_add_1d),
        ("二维矩阵加法", test_add_2d),
        ("ReLU 激活函数", test_relu),
        ("Sigmoid 激活函数", test_sigmoid),
        ("GELU 激活函数", test_gelu),
        ("Softmax 归约函数", test_softmax),
        ("Sum 归约函数", test_sum),
        ("Strided Add", test_strided_add),
        ("RMS Normalization", test_rms_norm),
    ]

    results = []
    start_time = time.time()

    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True, None))
        except Exception as e:
            print(f"❌ 测试 {name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
            print()

    end_time = time.time()

    # 打印总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed

    for name, success, error in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {name:20s}: {status}")

    print()
    print(f"通过: {passed}/{len(results)}")
    print(f"失败: {failed}/{len(results)}")
    print(f"耗时: {end_time - start_time:.2f} 秒")
    print()

    if failed == 0:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("❌ 部分测试失败，请检查上述错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
