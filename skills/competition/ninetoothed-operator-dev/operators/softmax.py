import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor


BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(x, output, BLOCK_SIZE=BLOCK_SIZE):
    return x.tile((1, BLOCK_SIZE)), output.tile((1, BLOCK_SIZE))


def application(x, output):
    x_max = ntl.max(x)
    x_shifted = x - x_max
    exp_x = ntl.exp(x_shifted)
    sum_exp = ntl.sum(exp_x)
    output = exp_x / sum_exp  # noqa: F841


def create_softmax_kernel():
    """
    创建 Softmax 算子内核（数值稳定版本）

    Softmax 将输入转换为概率分布，输出值在[0, 1]之间且和为1。
    本实现使用数值稳定技巧：减去最大值避免指数溢出。

    调用时需传入 BLOCK_SIZE 参数，值应等于输入的最后一维大小：
        kernel(x, output, BLOCK_SIZE=x.shape[-1])

    Returns:
        九齿内核函数
    """
    return ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2)))
