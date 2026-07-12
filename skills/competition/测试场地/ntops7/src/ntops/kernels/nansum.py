import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def application(input, output):
    dtype = output.dtype.dtype
    accumulator = ntl.cast(0, ntl.float32)
    for i in range(input.shape[0]):
        block = ntl.cast(input[i], ntl.float32)
        block = ntl.where(block != block, ntl.cast(0, ntl.float32), block)
        accumulator += ntl.sum(block, axis=0)
    output[0] = ntl.cast(accumulator, dtype)


def premake(ndim, dim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)
    tensors = (
        Tensor(ndim, dtype=dtype, other=0),
        Tensor(ndim, dtype=dtype),
    )
    return arrangement_, application, tensors


def arrangement_all_elements(input, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()
    input = input.flatten().tile((block_size,))
    output = output.tile((1,))
    return input, output


def application_all_elements(input, output):
    block = ntl.cast(input, ntl.float32)
    block = ntl.where(block != block, ntl.cast(0, ntl.float32), block)
    output[0] = ntl.sum(block, axis=0)


def premake_all_elements(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement_all_elements, block_size=block_size)
    tensors = (
        Tensor(ndim, dtype=dtype, other=0),
        Tensor(1, dtype=dtype),
    )
    return arrangement_, application_all_elements, tensors
