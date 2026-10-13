import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def default_application(input, other, output):
    output = input / other  # noqa: F841


def trunc_application(input, other, output):
    output = ntl.cast(input / other, ntl.int64)  # noqa: F841


def floor_application(input, other, output):
    output = ntl.floor(input / other)  # noqa: F841


def premake(ndim, rounding_mode, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    if rounding_mode == "trunc":
        application = trunc_application
    elif rounding_mode == "floor":
        application = floor_application
    else:
        application = default_application

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
