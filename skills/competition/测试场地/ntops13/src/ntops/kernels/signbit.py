import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    if input.dtype is ntl.float16:
        i_unint = ntl.cast(input, ntl.uint16, bitcast=True)
        output = (i_unint >> 15) & 0x1  # noqa: F841
    elif input.dtype is ntl.float32:
        i_unint = ntl.cast(input, ntl.uint32, bitcast=True)
        output = (i_unint >> 31) & 0x1  # noqa: F841
    elif input.dtype is ntl.float64:
        i_unint = ntl.cast(input, ntl.uint64, bitcast=True)
        output = (i_unint >> 63) & 0x1  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors
