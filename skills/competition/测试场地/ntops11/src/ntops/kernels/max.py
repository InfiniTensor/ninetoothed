import functools

import ninetoothed
import ninetoothed.language as ntl
import math
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def application(input, output, output_idx):
    # input: (C // block_size, ) dtype: (block_size, )
    # output: (C // block_size, ) dtype: (block_size, )
    dtype = output.dtype.dtype
    prev_max = ntl.cast(float("-inf"), dtype)
    global_idx = -1
    offset = input.dtype.shape[0]

    for i in range(input.shape[0]):
        curr_idx = ntl.argmax(input[i], 0) + (i * offset)
        curr_max = ntl.cast(ntl.maximum(prev_max, ntl.max(input[i])), dtype)
        global_idx = curr_idx if curr_max > prev_max else global_idx
        prev_max = curr_max

    output[0] = prev_max
    output_idx[0] = global_idx


def premake(ndim, dim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    tensors = (
        Tensor(
            ndim, dtype=dtype, other=float("-inf"), shape_options={"constexpr": True}
        ),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=ninetoothed.int32)
    )

    return arrangement_, application, tensors

def arrangement_all_elements(input, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input = input.flatten().tile((block_size,))
    output = output.tile((1,))
    return input, output

def application_all_elements(input, output):
    output[0] = ntl.max(input, 0)

def premake_all_elements(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement_all_elements, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype, other=float("-inf"), shape_options={"constexpr": True}),
        Tensor(1, dtype=dtype),
    )

    return arrangement_, application_all_elements, tensors
