"""
LeakyReLU 算子 — 正确参考实现（仅供评测对照，不提供给被评测 agent）
"""
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(x, output, BLOCK_SIZE=BLOCK_SIZE):
    return x.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))


def application(x, output):
    # LeakyReLU: f(x) = x if x > 0 else 0.01 * x
    # 使用 ntl.where 实现条件分支
    output = ntl.where(x > 0, x, 0.01 * x)  # noqa: F841


def create_leaky_relu_kernel():
    """创建 LeakyReLU 激活函数内核"""
    return ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))
