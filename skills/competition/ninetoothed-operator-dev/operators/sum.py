import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor


BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(x, output, BLOCK_SIZE=BLOCK_SIZE):
    return x.tile((BLOCK_SIZE,)), output.tile((1,))


def application(x, output):
    output = ntl.sum(x)  # noqa: F841


def create_sum_kernel():
    """
    创建 Sum 归约算子内核

    对输入张量进行求和归约操作，输出为标量。

    调用时需传入 BLOCK_SIZE 参数，值应等于输入的元素总数：
        kernel(x, output, BLOCK_SIZE=x.numel())

    Returns:
        九齿内核函数
    """
    return ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))
