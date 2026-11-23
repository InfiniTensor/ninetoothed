import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, tensor1, tensor2, value, output):
    # FC-15 fix: up-cast every operand to fp32, compute the full expression
    # in fp32, then cast back to the input dtype at the store.
    a = ntl.cast(input, ntl.float32)
    b = ntl.cast(tensor1, ntl.float32)
    c = ntl.cast(tensor2, ntl.float32)
    v = ntl.cast(value, ntl.float32)
    result = a + v * b * c
    output = result.to(input.dtype)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
