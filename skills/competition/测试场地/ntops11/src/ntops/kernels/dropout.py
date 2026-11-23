import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, p, seed, output):
    output = ntl.where(ntl.rand(seed, input.offsets()) > p, input / (1 - p), 0)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(0, dtype=ninetoothed.int64),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
