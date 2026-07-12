import functools
import math

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def default_application(input, output):
    output = input * 0.5 * (1 + ntl.erf(input / ntl.sqrt(2.0)))  # noqa: F841


def tanh_application(input, output):
    input_loaded = input

    output = (  # noqa: F841
        0.5
        * input_loaded
        * (
            1
            + ntl.tanh(
                ntl.sqrt(2 / math.pi) * (input_loaded + 0.044715 * input_loaded**3)
            )
        )
    )


def premake(ndim, approximate, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    if approximate == "tanh":
        application = tanh_application
    else:
        application = default_application

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors
