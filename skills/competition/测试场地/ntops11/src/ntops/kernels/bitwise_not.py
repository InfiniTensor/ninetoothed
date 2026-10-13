import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def bitwise_application(input, output):
    output = ~input  # noqa: F841


def logical_application(input, output):
    output = ntl.where(input, False, True)  # noqa: F841


def premake(ndim, logical=False, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    application = logical_application if logical else bitwise_application

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors
