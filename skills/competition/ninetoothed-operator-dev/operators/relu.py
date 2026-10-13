import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size


def create_relu_kernel():
    """
    创建 ReLU 激活函数内核

    ReLU (Rectified Linear Unit) 是最常用的激活函数之一，
    定义为 f(x) = max(0, x)

    Returns:
        九齿内核函数
    """
    BLOCK_SIZE = block_size()

    def arrangement(x, output):
        return x.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

    def application(x, output):
        output = ntl.maximum(x, 0.0)  # noqa: F841

    return ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))
