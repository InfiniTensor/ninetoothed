"""
九齿算子开发Skill - 使用示例

本文件展示如何使用已实现的算子。
注意：需要CUDA环境才能实际运行。

使用方式：
1. 直接运行此文件：python examples/usage_example.py
2. 导入到其他模块中使用
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def check_cuda():
    """检查CUDA是否可用"""
    if not torch.cuda.is_available():
        print("⚠️  警告：CUDA不可用，无法实际运行算子")
        print("   本示例将在CPU模式下验证语法正确性")
        return False
    return True


def example_gelu():
    """GELU算子使用示例"""
    print("\n" + "="*60)
    print("GELU 算子示例")
    print("="*60)

    from operators import create_gelu_kernel

    print("1. 导入成功：create_gelu_kernel")

    kernel = create_gelu_kernel()
    print("2. 创建内核成功")

    if not check_cuda():
        return

    x = torch.randn(1024, dtype=torch.float16, device='cuda')
    output = torch.empty_like(x)

    kernel(x, output)

    expected = torch.nn.functional.gelu(x)
    match = torch.allclose(output, expected, atol=1e-2, rtol=1e-2)

    print(f"3. 执行完成")
    print(f"4. PyTorch对比: {'通过 ✓' if match else '失败 ✗'}")


def example_softmax():
    """Softmax算子使用示例"""
    print("\n" + "="*60)
    print("Softmax 算子示例")
    print("="*60)

    from operators import create_softmax_kernel

    print("1. 导入成功：create_softmax_kernel")

    kernel = create_softmax_kernel()
    print("2. 创建内核成功")

    if not check_cuda():
        return

    x = torch.randn(512, 256, dtype=torch.float16, device='cuda')
    output = torch.empty_like(x)

    kernel(x, output, BLOCK_SIZE=x.shape[-1])

    expected = torch.softmax(x, dim=-1)
    match = torch.allclose(output, expected, atol=1e-2, rtol=1e-2)

    print(f"3. 执行完成")
    print(f"4. PyTorch对比: {'通过 ✓' if match else '失败 ✗'}")


def example_add():
    """Add算子使用示例"""
    print("\n" + "="*60)
    print("Add 算子示例")
    print("="*60)

    from operators import create_add_kernel

    print("1. 导入成功：create_add_kernel")

    kernel = create_add_kernel()
    print("2. 创建内核成功")

    if not check_cuda():
        return

    a = torch.randn(1024, dtype=torch.float16, device='cuda')
    b = torch.randn(1024, dtype=torch.float16, device='cuda')
    c = torch.empty_like(a)

    kernel(a, b, c)

    expected = a + b
    match = torch.allclose(c, expected, atol=1e-5, rtol=1e-3)

    print(f"3. 执行完成")
    print(f"4. PyTorch对比: {'通过 ✓' if match else '失败 ✗'}")


def example_all():
    """所有算子使用示例"""
    print("\n" + "="*60)
    print("所有算子示例")
    print("="*60)

    from operators import (
        create_gelu_kernel,
        create_softmax_kernel,
        create_relu_kernel,
        create_sigmoid_kernel,
        create_add_kernel,
        create_sum_kernel,
        create_strided_add_kernel,
        create_rms_norm_kernel,
    )

    operators = [
        ("GELU", create_gelu_kernel),
        ("Softmax", create_softmax_kernel),
        ("ReLU", create_relu_kernel),
        ("Sigmoid", create_sigmoid_kernel),
        ("Add", create_add_kernel),
        ("Sum", create_sum_kernel),
        ("Strided Add", create_strided_add_kernel),
        ("RMS Norm", create_rms_norm_kernel),
    ]

    if not torch.cuda.is_available():
        print("\n⚠️  CUDA不可用，仅验证导入成功")
        print("   所有算子函数导入成功")
        return

    for name, create_func in operators:
        print(f"\n创建 {name} 算子...")
        kernel = create_func()
        print(f"  ✓ {name} 算子创建成功")


if __name__ == "__main__":
    print("="*60)
    print("九齿算子开发Skill - 使用示例")
    print("="*60)

    example_all()

    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("CUDA可用，运行实际测试")
        print("="*60)

        example_gelu()
        example_softmax()
        example_add()
    else:
        print("\n⚠️  CUDA不可用，仅展示语法验证")
        print("   如需实际运行，请在有GPU的环境中执行")

    print("\n" + "="*60)
    print("示例完成")
    print("="*60)
