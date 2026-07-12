import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size


def create_sigmoid_kernel():
    """
    创建 Sigmoid 激活函数内核

    Sigmoid 函数将输入映射到(0, 1)区间，常用于二分类问题。
    定义为 f(x) = 1 / (1 + exp(-x))

    Returns:
        九齿内核函数
    """
    BLOCK_SIZE = block_size()

    def arrangement(x, output):
        return x.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))

    def application(x, output):
        output = ntl.sigmoid(ntl.cast(x, ntl.float32))  # noqa: F841

    return ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))
