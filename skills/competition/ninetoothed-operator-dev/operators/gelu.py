import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size


def create_gelu_kernel():
    """
    创建 GELU 激活函数内核

    GELU (Gaussian Error Linear Units) 是一种平滑的激活函数，
    在Transformer架构中被广泛使用。

    Returns:
        九齿内核函数
    """
    BLOCK_SIZE = block_size()

    def arrangement(x, output):
        return x.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

    def application(x, output):
        x_f32 = ntl.cast(x, ntl.float32)
        # sqrt(2 / pi) 预计算
        sqrt_2_over_pi = 0.7978845608028654
        inner = sqrt_2_over_pi * (x_f32 + 0.044715 * x_f32 * x_f32 * x_f32)
        # tanh(inner) = 2 * sigmoid(2 * inner) - 1
        tanh_inner = 2.0 * ntl.sigmoid(2.0 * inner) - 1.0
        cdf = 0.5 * (1.0 + tanh_inner)
        output = x_f32 * cdf  # noqa: F841

    return ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))
