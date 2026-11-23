"""
Strided Add 算子 —— 非连续输入 + 步长场景。

演示如何通过 Tensor 切片（__getitem__）处理带步长的非连续张量访问。
参考 ninetoothed 仓库 tests/test_getitem.py 和 tests/test_pad.py 中的 stride/slice 模式。

1D: output[::2] = input[::2] + other[::2]
2D: output[:, ::2] = input[:, ::2] + other[:, ::2]  (沿最后一维步长)
"""
import ninetoothed
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
BLOCK_SIZE_ROW = Symbol("BLOCK_SIZE_ROW", constexpr=True)
BLOCK_SIZE_COL = Symbol("BLOCK_SIZE_COL", constexpr=True)


# ========== 1D 版本 ==========


def arrangement_1d(input, other, output, BLOCK_SIZE=BLOCK_SIZE):
    input_strided = input[::2]
    other_strided = other[::2]
    output_strided = output[::2]

    return (
        input_strided.tile((BLOCK_SIZE,)),
        other_strided.tile((BLOCK_SIZE,)),
        output_strided.tile((BLOCK_SIZE,)),
    )


def application_1d(input, other, output):
    output = input + other  # noqa: F841


tensors_1d = (Tensor(1), Tensor(1), Tensor(1))


def create_strided_add_kernel():
    """Strided Add kernel — 1D 版本

    用法:
        kernel = create_strided_add_kernel()
        c = torch.zeros_like(a)
        kernel(a, b, c, BLOCK_SIZE=a.shape[0])
        # 效果: c[::2] = a[::2] + b[::2]
    """
    return ninetoothed.make(arrangement_1d, application_1d, tensors_1d)


# ========== 2D 版本 ==========


def arrangement_2d(input, other, output,
                   BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
                   BLOCK_SIZE_COL=BLOCK_SIZE_COL):
    """
    沿最后一维做 stride=2，第一维正常分块。
    非连续访存体现在列方向上，行方向保持合并访问。
    """
    input_strided = input[:, ::2]
    other_strided = other[:, ::2]
    output_strided = output[:, ::2]

    return (
        input_strided.tile((BLOCK_SIZE_ROW, BLOCK_SIZE_COL)),
        other_strided.tile((BLOCK_SIZE_ROW, BLOCK_SIZE_COL)),
        output_strided.tile((BLOCK_SIZE_ROW, BLOCK_SIZE_COL)),
    )


def application_2d(input, other, output):
    output = input + other  # noqa: F841


tensors_2d = (Tensor(2), Tensor(2), Tensor(2))


def create_2d_strided_add_kernel():
    """Strided Add kernel — 2D 版本（沿列方向步长）

    用法:
        kernel = create_2d_strided_add_kernel()
        c = torch.zeros_like(a)
        kernel(a, b, c, BLOCK_SIZE_ROW=a.shape[0], BLOCK_SIZE_COL=a.shape[1] // 2)
        # 效果: c[:, ::2] = a[:, ::2] + b[:, ::2]
    """
    return ninetoothed.make(arrangement_2d, application_2d, tensors_2d)
