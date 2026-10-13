import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def application(input, weight, eps, output, num_normalized_elements):
    _rms = ntl.zeros(input.dtype.shape, dtype=ntl.float32)

    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        _rms += input_i * input_i

    rms = ntl.sqrt(ntl.sum(_rms) / num_normalized_elements + eps)

    for i in range(input.shape[0]):
        output[i] = input[i] / rms * weight[i]


def premake(
    ndim,
    num_normalized_dims,
    input_dtype=None,
    weight_dtype=None,
    output_dtype=None,
    block_size=None,
):
    dims = tuple(-(dim + 1) for dim in range(num_normalized_dims))

    arrangement_ = functools.partial(arrangement, dim=dims, block_size=block_size)

    tensors = (
        Tensor(ndim, other=0, dtype=input_dtype),
        Tensor(ndim, dtype=weight_dtype),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(ndim, dtype=output_dtype),
        Tensor(0, dtype=ninetoothed.int64),
    )

    return arrangement_, application, tensors
