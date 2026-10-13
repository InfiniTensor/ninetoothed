import copy
import functools

import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

from ntops.kernels import mm


def arrangement(
    input,
    weight,
    bias,
    output,
    input_precision,
    stride_h=None,
    stride_w=None,
    padding_h=None,
    padding_w=None,
    dilation_h=None,
    dilation_w=None,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    if stride_h is None:
        stride_h = Symbol("stride_h", constexpr=True)

    if stride_w is None:
        stride_w = Symbol("stride_w", constexpr=True)

    if padding_h is None:
        padding_h = Symbol("padding_h", constexpr=True)

    if padding_w is None:
        padding_w = Symbol("padding_w", constexpr=True)

    if dilation_h is None:
        dilation_h = Symbol("dilation_h", constexpr=True)

    if dilation_w is None:
        dilation_w = Symbol("dilation_w", constexpr=True)

    if block_size_m is None:
        block_size_m = mm.BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = mm.BLOCK_SIZE_N

    if block_size_k is None:
        block_size_k = mm.BLOCK_SIZE_K

    mm_arrangement = functools.partial(
        mm.arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    input_arranged = input.pad(
        ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w))
    )
    input_arranged = input_arranged.tile(
        (1, *weight.shape[1:]),
        strides=(-1, -1, stride_h, stride_w),
        dilation=(1, 1, dilation_h, dilation_w),
        floor_mode=True,
    )
    input_arranged = input_arranged.squeeze(1)
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=3).flatten(start_dim=1)

    weight_arranged = weight.flatten(start_dim=1)
    weight_arranged = weight_arranged.permute((1, 0))

    bias_arranged = bias[None, :, None, None].expand(
        (output.shape[0], -1, output.shape[2], output.shape[3])
    )
    bias_arranged = bias_arranged.permute((0, 2, 3, 1)).flatten(end_dim=3)

    output_arranged = output.permute((0, 2, 3, 1)).flatten(end_dim=3)

    _, _, bias_arranged, _ = mm_arrangement(
        copy.deepcopy(input_arranged),
        copy.deepcopy(weight_arranged),
        bias_arranged,
        copy.deepcopy(input_precision),
    )

    input_arranged, weight_arranged, output_arranged, input_precision_arranged = (
        mm_arrangement(
            input_arranged, weight_arranged, output_arranged, input_precision
        )
    )

    return (
        input_arranged,
        weight_arranged,
        bias_arranged,
        output_arranged,
        input_precision_arranged,
    )


def application(input, weight, bias, output, input_precision):
    mm_output = ntl.zeros(output.shape, dtype=ntl.float32)
    mm.application(input, weight, mm_output, input_precision)
    output = mm_output + bias


def premake(
    input_precision=None,
    stride_h=None,
    stride_w=None,
    padding_h=None,
    padding_w=None,
    dilation_h=None,
    dilation_w=None,
    dtype=None,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    arrangement_ = functools.partial(
        arrangement,
        stride_h=stride_h,
        stride_w=stride_w,
        padding_h=padding_h,
        padding_w=padding_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    input, weight, output = (Tensor(4, dtype=dtype) for _ in range(3))
    bias = Tensor(1, dtype=dtype)
    input_precision = Tensor(0, dtype=dtype, constexpr=True, value=input_precision)

    tensors = (input, weight, bias, output, input_precision)

    return arrangement_, application, tensors
