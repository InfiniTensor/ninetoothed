import functools

import ninetoothed
from ninetoothed import Symbol, Tensor


def arrangement_embed(input, output, stride=None, block_size=None):
    if stride is None:
        stride = Symbol("stride", constexpr=True)

    if block_size is None:
        block_size = ninetoothed.block_size()

    input_arranged = input.tile((block_size,))
    output_arranged = output.tile(
        (block_size,), strides=(block_size * stride,), dilation=(stride,)
    )

    return input_arranged, output_arranged


def arrangement_extract(input, output, stride=None, block_size=None):
    if stride is None:
        stride = Symbol("stride", constexpr=True)

    if block_size is None:
        block_size = ninetoothed.block_size()

    input_arranged = input.tile(
        (block_size,), strides=(block_size * stride,), dilation=(stride,)
    )
    output_arranged = output.tile((block_size,))

    return input_arranged, output_arranged


def application(input, output):
    output = input  # noqa: F841


def premake_embed(stride=None, dtype=None, block_size=None):
    arrangement_ = functools.partial(
        arrangement_embed, stride=stride, block_size=block_size
    )

    tensors = (Tensor(1, dtype=dtype, other=0), Tensor(1, dtype=dtype))

    return arrangement_, application, tensors


def premake_extract(stride=None, dtype=None, block_size=None):
    arrangement_ = functools.partial(
        arrangement_extract, stride=stride, block_size=block_size
    )

    tensors = (Tensor(1, dtype=dtype, other=0), Tensor(1, dtype=dtype))

    return arrangement_, application, tensors
