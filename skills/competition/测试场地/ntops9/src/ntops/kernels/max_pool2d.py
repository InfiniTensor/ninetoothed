import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.pooling import arrangement


def application(input, output):
    output = ntl.max(input, axis=-1)  # noqa: F841


def premake(
    kernel_size_h=None,
    kernel_size_w=None,
    stride_h=None,
    stride_w=None,
    padding_h=None,
    padding_w=None,
    dilation_h=None,
    dilation_w=None,
    ceil_mode=None,
    dtype=None,
    block_size=None,
):
    arrangement_ = functools.partial(
        arrangement,
        kernel_size_h=kernel_size_h,
        kernel_size_w=kernel_size_w,
        stride_h=stride_h,
        stride_w=stride_w,
        padding_h=padding_h,
        padding_w=padding_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        ceil_mode=ceil_mode,
        block_size=block_size,
    )

    tensors = (Tensor(4, dtype=dtype, other=float("-inf")), Tensor(4, dtype=dtype))

    return arrangement_, application, tensors
