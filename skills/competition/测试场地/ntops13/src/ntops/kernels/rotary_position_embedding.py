import functools

from ninetoothed import Tensor


def arrangement(input, sin_table, cos_table, output, interleaved=True):
    emb_dim = input.shape[-1]
    tile_shape = (1, 1, 1, emb_dim // 2)

    if interleaved:
        strides = (-1, -1, -1, 1)
        dilation = (1, 1, 1, 2)
    else:
        strides = None
        dilation = None

    def _arrange_input_or_output(tensor):
        tensor_arranged = tensor.tile(tile_shape, strides=strides, dilation=dilation)
        tensor_arranged = tensor_arranged.tile((1, 1, 1, -1))
        tensor_arranged.dtype = tensor_arranged.dtype.squeeze((0, 1, 2))
        tensor_arranged.dtype.dtype = tensor_arranged.dtype.dtype.squeeze((0, 1, 2))

        return tensor_arranged

    def _arrange_table(table):
        table_arranged = table.tile(tile_shape)
        table_arranged.dtype = table_arranged.dtype.squeeze((0, 1, 2))

        return table_arranged

    input_arranged = _arrange_input_or_output(input)
    sin_table_arranged = _arrange_table(sin_table)
    cos_table_arranged = _arrange_table(cos_table)
    output_arranged = _arrange_input_or_output(output)

    return input_arranged, sin_table_arranged, cos_table_arranged, output_arranged


def application(input, sin_table, cos_table, output):
    sin_table_loaded = sin_table
    cos_table_loaded = cos_table

    input_0 = input[0]
    input_1 = input[1]

    output[0] = input_0 * cos_table_loaded - input_1 * sin_table_loaded
    output[1] = input_0 * sin_table_loaded + input_1 * cos_table_loaded


def premake(ndim, emb_dim=None, dtype=None, interleaved=True):
    arrangement_ = functools.partial(arrangement, interleaved=interleaved)

    shape_options = (None, None, None, {"constexpr": True, "upper_bound": 128})

    tensors = (
        Tensor(ndim, dtype=dtype, shape_options=shape_options),
        Tensor(ndim, dtype=dtype, shape_options=shape_options),
        Tensor(ndim, dtype=dtype, shape_options=shape_options),
        Tensor(ndim, dtype=dtype, shape_options=shape_options),
    )

    if emb_dim is not None:
        for tensor in tensors:
            tensor.shape = tensor.shape[:-1] + (emb_dim,)

    return arrangement_, application, tensors
