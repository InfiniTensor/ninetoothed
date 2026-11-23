"""
九齿算子模板文件

使用说明：
1. 将此文件复制为新算子文件
2. 替换算子名称和逻辑
3. 使用 ninetoothed.make() 构建内核
4. 添加必要的测试

九齿使用 arrange-and-apply 范式：
- arrangement 函数：定义张量的排列方式（编译时）
- application 函数：定义如何应用排列后的张量（运行时）
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size


def create_operator_name_kernel():
    """
    创建 operator_name 算子内核

    Returns:
        九齿内核函数
    """
    BLOCK_SIZE = block_size()

    def arrangement(x, output):
        return x.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

    def application(x, output):
        output = x * 2

    return ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))


def create_add_kernel():
    """
    创建向量加法算子内核

    Returns:
        九齿内核函数
    """
    BLOCK_SIZE = block_size()

    def arrangement(x, y, output):
        return x.tile((BLOCK_SIZE,)), y.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

    def application(x, y, output):
        output = x + y

    return ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)))


def create_softmax_kernel():
    """
    创建 Softmax 算子内核（数值稳定版本）

    注意：归约算子需要使用 Symbol(constexpr=True) 而非 block_size()
    以避免 BLOCK_SIZE 小于归约维度导致局部归约错误。

    调用时需传入 BLOCK_SIZE 参数：
        kernel(x, output, BLOCK_SIZE=x.shape[-1])

    Returns:
        九齿内核函数
    """
    from ninetoothed import Symbol
    BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

    def arrangement(x, output, BLOCK_SIZE=BLOCK_SIZE):
        return x.tile((1, BLOCK_SIZE)), output.tile((1, BLOCK_SIZE))

    def application(x, output):
        x_max = ntl.max(x)
        x_shifted = x - x_max
        exp_x = ntl.exp(x_shifted)
        sum_exp = ntl.sum(exp_x)
        output = exp_x / sum_exp

    return ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2)))


def create_gelu_kernel():
    """
    创建 GELU 激活函数内核

    Returns:
        九齿内核函数
    """
    BLOCK_SIZE = block_size()

    def arrangement(x, output):
        return x.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

    def application(x, output):
        x_f32 = ntl.cast(x, ntl.float32)
        sqrt_2_over_pi = 0.7978845608028654
        inner = sqrt_2_over_pi * (x_f32 + 0.044715 * x_f32 * x_f32 * x_f32)
        tanh_inner = 2.0 * ntl.sigmoid(2.0 * inner) - 1.0
        cdf = 0.5 * (1.0 + tanh_inner)
        output = x_f32 * cdf

    return ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))


if __name__ == "__main__":
    import torch

    print("=== 九齿算子模板测试 ===\n")

    print("1. 测试 operator_name 算子:")
    kernel = create_operator_name_kernel()
    x = torch.randn(1024, dtype=torch.float16, device='cuda')
    output = torch.empty_like(x)
    kernel(x, output)
    print(f"   输入形状: {x.shape}, 输出形状: {output.shape}")
    print(f"   输出统计: min={output.min().item():.4f}, max={output.max().item():.4f}")

    print("\n2. 测试 add 算子:")
    add_kernel = create_add_kernel()
    a = torch.randn(1024, dtype=torch.float16, device='cuda')
    b = torch.randn(1024, dtype=torch.float16, device='cuda')
    c = torch.empty_like(a)
    add_kernel(a, b, c)
    expected = a + b
    match = torch.allclose(c, expected, atol=1e-2, rtol=1e-2)
    print(f"   PyTorch 对比: {'通过 ✓' if match else '失败 ✗'}")

    print("\n3. 测试 softmax 算子:")
    softmax_kernel = create_softmax_kernel()
    x = torch.randn(512, dtype=torch.float16, device='cuda')
    output = torch.empty_like(x)
    softmax_kernel(x, output)
    expected = torch.softmax(x, dim=-1)
    match = torch.allclose(output, expected, atol=1e-2, rtol=1e-2)
    print(f"   PyTorch 对比: {'通过 ✓' if match else '失败 ✗'}")

    print("\n4. 测试 GELU 算子:")
    gelu_kernel = create_gelu_kernel()
    x = torch.randn(1024, dtype=torch.float16, device='cuda')
    output = torch.empty_like(x)
    gelu_kernel(x, output)
    expected = torch.nn.functional.gelu(x)
    match = torch.allclose(output, expected, atol=1e-2, rtol=1e-2)
    print(f"   PyTorch 对比: {'通过 ✓' if match else '失败 ✗'}")

    print("\n=== 测试完成 ===")
