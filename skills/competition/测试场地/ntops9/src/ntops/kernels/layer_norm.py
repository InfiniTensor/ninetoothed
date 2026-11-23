import functools
import math

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def application(input, weight, bias, eps, output, num_normalized_elements):
    _mean = ntl.zeros(input.dtype.shape, dtype=ntl.float32)

    for i in range(input.shape[0]):
        _mean += ntl.cast(input[i], ntl.float32)

    mean = ntl.sum(_mean, 0) / num_normalized_elements

    _var = ntl.zeros(input.dtype.shape, dtype=ntl.float32)

    for i in range(input.shape[0]):
        diff = ntl.cast(input[i], ntl.float32) - mean
        diff = ntl.where(input[i].offsets(-1) < input.source.shape[-1], diff, 0)
        _var += diff * diff

    var = ntl.sum(_var, 0) / num_normalized_elements

    std = ntl.sqrt(var + eps)

    for i in range(input.shape[0]):
        output[i] = (ntl.cast(input[i], ntl.float32) - mean) / std * weight[i] + bias[i]


def premake(ndim, normalized_shape, dtype=None, block_size=None):
    dims = tuple(-(dim + 1) for dim in range(len(normalized_shape)))

    arrangement_ = functools.partial(arrangement, dim=dims, block_size=block_size)

    tensors = (
        Tensor(ndim, other=0, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(ndim, dtype=dtype),
        Tensor(0, constexpr=True, value=math.prod(normalized_shape)),
    )

    return arrangement_, application, tensors
