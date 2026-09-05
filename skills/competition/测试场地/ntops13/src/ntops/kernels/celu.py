import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, alpha, output):
    output = max(0.0, input) + min(0.0, alpha * (ntl.exp(input / alpha) - 1))  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
