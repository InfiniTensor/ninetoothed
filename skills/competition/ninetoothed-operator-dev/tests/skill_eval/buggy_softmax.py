"""
BUGGY Softmax — 故意引入的错误版本

错误：tile 形状使用 (BLOCK_SIZE, BLOCK_SIZE) 而非正确的 (1, BLOCK_SIZE)
      导致同一行被拆分到多个 block，每个 block 独立计算 exp/softmax，
      结果错误（约 50% 元素不匹配）。
"""
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor


BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(x, output, BLOCK_SIZE=BLOCK_SIZE):
    # BUG: 归约算子应该用 (1, BLOCK_SIZE)，错误地用了 (BLOCK_SIZE, BLOCK_SIZE)
    return x.tile((BLOCK_SIZE, BLOCK_SIZE)), output.tile((BLOCK_SIZE, BLOCK_SIZE))


def application(x, output):
    x_max = ntl.max(x)
    x_shifted = x - x_max
    exp_x = ntl.exp(x_shifted)
    sum_exp = ntl.sum(exp_x)
    output = exp_x / sum_exp  # noqa: F841


def create_softmax_kernel():
    return ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2)))
