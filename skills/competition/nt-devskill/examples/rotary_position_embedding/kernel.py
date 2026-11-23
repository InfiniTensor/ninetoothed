import functools

import ninetoothed
from ninetoothed import Tensor


def arrangement(input, sin_table, cos_table, interleaved=True):
    emb_dim = input.shape[-1]
    tile_shape = (1, 1, 1, emb_dim // 2)

    if interleaved:
        strides = (-1, -1, -1, 1)
        dilation = (1, 1, 1, 2)
    else:
        strides = None
        dilation = None

    input_arranged = input.tile(tile_shape, strides=strides, dilation=dilation)
    input_arranged = input_arranged.tile((1, 1, 1, 2))
    input_arranged.dtype = input_arranged.dtype.squeeze((0, 1, 2))
    input_arranged.dtype.dtype = input_arranged.dtype.dtype.squeeze((0, 1, 2))

    sin_table_arranged = sin_table.tile(tile_shape)
    sin_table_arranged.dtype = sin_table_arranged.dtype.squeeze((0, 1, 2))

    cos_table_arranged = cos_table.tile(tile_shape)
    cos_table_arranged.dtype = cos_table_arranged.dtype.squeeze((0, 1, 2))

    return input_arranged, sin_table_arranged, cos_table_arranged


def application(input, sin_table, cos_table):
    # RoPE: 旋转位置编码
    input_0 = input[0]
    input_1 = input[1]

    input[0] = input_0 * cos_table - input_1 * sin_table
    input[1] = input_0 * sin_table + input_1 * cos_table


inputs = tuple(Tensor(4, shape_options={"constexpr": True}) for _ in range(3))

interleaved_kernel = ninetoothed.make(
    functools.partial(arrangement, interleaved=True), application, inputs
)
non_interleaved_kernel = ninetoothed.make(
    functools.partial(arrangement, interleaved=False), application, inputs
)


def kernel(input, sin_table, cos_table, interleaved=True):
    return (interleaved_kernel if interleaved else non_interleaved_kernel)(
        input, sin_table, cos_table
    )
