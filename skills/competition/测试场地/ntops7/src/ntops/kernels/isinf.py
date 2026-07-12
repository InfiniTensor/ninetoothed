import functools

from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    pos_result = input == float("+inf")
    neg_result = input == float("-inf")
    output = pos_result or neg_result  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors
