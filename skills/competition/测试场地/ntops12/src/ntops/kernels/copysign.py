import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, other, output):
    if input.dtype is ntl.float16:
        other_bits = ntl.cast(other, ntl.uint16, bitcast=True)
        other_sign = other_bits >> 15
    elif input.dtype is ntl.float32:
        other_bits = ntl.cast(other, ntl.uint32, bitcast=True)
        other_sign = other_bits >> 31

    abs_input = ntl.abs(input)

    output = ntl.where(other_sign > 0, -abs_input, abs_input)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
