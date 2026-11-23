"""
九齿算子测试模板文件

使用说明：
1. 将此文件复制为新测试文件
2. 替换算子创建函数
3. 实现测试用例
4. 运行测试验证算子正确性

九齿算子使用 arrange-and-apply 范式，需要使用 ninetoothed.make() 构建内核。
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_kernel():
    """
    创建测试算子内核

    此函数应替换为实际的算子创建逻辑

    Returns:
        九齿内核函数
    """
    import ninetoothed
    import ninetoothed.language as ntl
    from ninetoothed import Tensor, Symbol

    BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

    def arrangement(x, output, BLOCK_SIZE=128):
        return x.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

    def application(x, output):
        output = x * 2

    return ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))


class TestOperatorKernel:
    """九齿算子测试类"""

    def test_basic_functionality(self):
        """测试基本功能"""
        kernel = create_test_kernel()

        input_tensor = torch.randn(1024, dtype=torch.float16, device='cuda')
        output = torch.empty_like(input_tensor)

        kernel(input_tensor, output)

        expected = input_tensor * 2
        assert output.shape == input_tensor.shape
        assert torch.allclose(output, expected, atol=1e-2, rtol=1e-2)

    def test_multiple_shapes(self):
        """测试多种输入形状"""
        kernel = create_test_kernel()

        test_shapes = [
            (100,),
            (512,),
            (1024,),
            (32, 64),
            (16, 32, 64),
        ]

        for shape in test_shapes:
            input_tensor = torch.randn(shape, dtype=torch.float16, device='cuda')
            output = torch.empty_like(input_tensor)

            kernel(input_tensor, output)

            expected = input_tensor * 2
            assert output.shape == input_tensor.shape
            assert torch.allclose(output, expected, atol=1e-2, rtol=1e-2)

    def test_different_dtypes(self):
        """测试不同数据类型"""
        kernel = create_test_kernel()

        test_dtypes = [
            (torch.float16, 1e-2, 1e-2),
            (torch.float32, 1e-5, 1e-3),
        ]

        for dtype, atol, rtol in test_dtypes:
            input_tensor = torch.randn(1024, dtype=dtype, device='cuda')
            output = torch.empty_like(input_tensor)

            kernel(input_tensor, output)

            expected = input_tensor * 2
            assert output.dtype == dtype
            assert torch.allclose(output, expected, atol=atol, rtol=rtol)

    def test_boundary_conditions(self):
        """测试边界条件"""
        kernel = create_test_kernel()

        zero_input = torch.zeros(1024, dtype=torch.float16, device='cuda')
        output_zero = torch.empty_like(zero_input)
        kernel(zero_input, output_zero)
        expected_zero = zero_input * 2
        assert torch.allclose(output_zero, expected_zero, atol=1e-2, rtol=1e-2)

        extreme_input = torch.tensor([0.0, 1.0, -1.0, 100.0, -100.0],
                                    dtype=torch.float16, device='cuda')
        output_extreme = torch.empty_like(extreme_input)
        kernel(extreme_input, output_extreme)
        expected_extreme = extreme_input * 2
        assert torch.allclose(output_extreme, expected_extreme, atol=1e-2, rtol=1e-2)

    def test_non_contiguous_input(self):
        """测试非连续输入"""
        kernel = create_test_kernel()

        base_tensor = torch.randn(1024, dtype=torch.float16, device='cuda')
        non_contiguous_input = base_tensor[::2]

        assert not non_contiguous_input.is_contiguous()

        output = torch.empty_like(non_contiguous_input)
        kernel(non_contiguous_input, output)

        expected = non_contiguous_input * 2
        assert output.shape == non_contiguous_input.shape
        assert torch.allclose(output, expected, atol=1e-2, rtol=1e-2)

    def test_pytorch_comparison(self):
        """与PyTorch实现对比"""
        kernel = create_test_kernel()

        input_tensor = torch.randn(1024, dtype=torch.float16, device='cuda')
        output = torch.empty_like(input_tensor)

        kernel(input_tensor, output)

        expected = input_tensor * 2
        assert torch.allclose(output, expected, atol=1e-2, rtol=1e-2)


class TestAddKernel:
    """向量加法算子测试"""

    @staticmethod
    def create_add_kernel(BLOCK_SIZE=128):
        """创建 add 算子内核"""
        import ninetoothed
        import ninetoothed.language as ntl
        from ninetoothed import Tensor, Symbol

        BLOCK_SIZE_VAL = Symbol("BLOCK_SIZE_VAL", constexpr=True)

        def arrangement(x, y, output, BLOCK_SIZE_VAL=BLOCK_SIZE_VAL):
            return (x.tile((BLOCK_SIZE_VAL,)),
                   y.tile((BLOCK_SIZE_VAL,)),
                   output.tile((BLOCK_SIZE_VAL,)))

        def application(x, y, output):
            output = x + y

        return ninetoothed.make(arrangement, application,
                              (Tensor(1), Tensor(1), Tensor(1)))

    def test_add_basic(self):
        """测试基本加法"""
        kernel = self.create_add_kernel()

        a = torch.randn(1024, dtype=torch.float16, device='cuda')
        b = torch.randn(1024, dtype=torch.float16, device='cuda')
        c = torch.empty_like(a)

        kernel(a, b, c)

        expected = a + b
        assert torch.allclose(c, expected, atol=1e-2, rtol=1e-2)


class TestSoftmaxKernel:
    """Softmax算子测试"""

    @staticmethod
    def create_softmax_kernel(BLOCK_SIZE=128):
        """创建 softmax 算子内核"""
        import ninetoothed
        import ninetoothed.language as ntl
        from ninetoothed import Tensor, Symbol

        BLOCK_SIZE_VAL = Symbol("BLOCK_SIZE_VAL", constexpr=True)

        def arrangement(x, output, BLOCK_SIZE_VAL=BLOCK_SIZE_VAL):
            return x.tile((1, BLOCK_SIZE_VAL)), output.tile((1, BLOCK_SIZE_VAL))

        def application(x, output):
            x_max = ntl.max(x)
            x_shifted = x - x_max
            exp_x = ntl.exp(x_shifted)
            sum_exp = ntl.sum(exp_x)
            output = exp_x / sum_exp

        return ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2)))

    def test_softmax_basic(self):
        """测试基本 Softmax"""
        kernel = self.create_softmax_kernel()

        x = torch.randn(512, dtype=torch.float16, device='cuda')
        output = torch.empty_like(x)

        kernel(x, output)

        expected = torch.softmax(x, dim=-1)
        assert torch.allclose(output, expected, atol=1e-2, rtol=1e-2)

    def test_softmax_sum_to_one(self):
        """测试 Softmax 输出和为1"""
        kernel = self.create_softmax_kernel()

        x = torch.randn(512, dtype=torch.float16, device='cuda')
        output = torch.empty_like(x)

        kernel(x, output)

        sum_output = output.sum(-1)
        assert torch.allclose(sum_output, torch.ones_like(sum_output), atol=1e-5, rtol=1e-5)

    def test_softmax_numerical_stability(self):
        """测试 Softmax 数值稳定性"""
        kernel = self.create_softmax_kernel()

        extreme_x = torch.tensor([[1000.0, 1001.0, 1002.0]],
                               dtype=torch.float16, device='cuda')
        output = torch.empty_like(extreme_x)

        kernel(extreme_x, output)

        assert torch.isfinite(output).all()
        sum_output = output.sum(-1)
        assert torch.allclose(sum_output, torch.ones_like(sum_output), atol=1e-5, rtol=1e-5)


class TestGELUKernel:
    """GELU算子测试"""

    @staticmethod
    def create_gelu_kernel(BLOCK_SIZE=128):
        """创建 GELU 算子内核"""
        import ninetoothed
        import ninetoothed.language as ntl
        from ninetoothed import Tensor, Symbol

        BLOCK_SIZE_VAL = Symbol("BLOCK_SIZE_VAL", constexpr=True)

        def arrangement(x, output, BLOCK_SIZE_VAL=BLOCK_SIZE_VAL):
            return x.tile((BLOCK_SIZE_VAL,)), output.tile((BLOCK_SIZE_VAL,))

        def application(x, output):
            x_f32 = ntl.cast(x, ntl.float32)
            sqrt_2_over_pi = 0.7978845608028654
            inner = sqrt_2_over_pi * (x_f32 + 0.044715 * x_f32 * x_f32 * x_f32)
            tanh_inner = 2.0 * ntl.sigmoid(2.0 * inner) - 1.0
            cdf = 0.5 * (1.0 + tanh_inner)
            output = x_f32 * cdf

        return ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))

    def test_gelu_basic(self):
        """测试基本 GELU"""
        kernel = self.create_gelu_kernel()

        x = torch.randn(1024, dtype=torch.float16, device='cuda')
        output = torch.empty_like(x)

        kernel(x, output)

        expected = torch.nn.functional.gelu(x)
        assert torch.allclose(output, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
