import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)
WINDOW_HEIGHT = Symbol("WINDOW_HEIGHT", constexpr=True, upper_bound=16)
WINDOW_WIDTH = Symbol("WINDOW_WIDTH", constexpr=True, upper_bound=16)


def arrangement(input, output):
    # 按窗口 tile input: (1, 1, WH, WW)
    input_arranged = input.tile((1, 1, WINDOW_HEIGHT, WINDOW_WIDTH))
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=4).flatten(start_dim=1)
    input_arranged = input_arranged.tile((BLOCK_SIZE, -1))

    output_arranged = output.tile((1, 1, 1, 1))
    output_arranged = output_arranged.ravel()
    output_arranged = output_arranged.flatten(end_dim=4).flatten(start_dim=1)
    output_arranged = output_arranged.tile((BLOCK_SIZE, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)

    return input_arranged, output_arranged


def application(input, output):
    # 对窗口内所有元素取 max
    output = ntl.max(input, axis=1)


kernel = ninetoothed.make(
    arrangement, application, (Tensor(4, other=float("-inf")), Tensor(4))
)
