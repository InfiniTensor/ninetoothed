"""
HardSwish 2D — 正确参考实现（仅供评测对照，不提供给被评测 agent）

HardSwish: f(x) = x * clamp(x + 3, 0, 6) / 6

此测试验证 agent 能否：
1. 独立推断 2D 逐元素 tile 策略（skill 无 2D elementwise 直接范例）
2. 正确设置 2D Symbol（模块级 + 参数默认值）
3. 编写 2D correctness 测试
"""
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor


BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", constexpr=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", constexpr=True)


def arrangement(x, output, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N):
    return (x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
            output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N)))


def application(x, output):
    # HardSwish: output = x * clamp(x + 3, 0, 6) / 6
    x_plus_3 = x + 3.0
    clamped = ntl.minimum(ntl.maximum(x_plus_3, 0.0), 6.0)
    output = x * clamped / 6.0  # noqa: F841


def create_hardswish_2d_kernel():
    """创建 HardSwish 2D 激活函数内核"""
    return ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2)))
