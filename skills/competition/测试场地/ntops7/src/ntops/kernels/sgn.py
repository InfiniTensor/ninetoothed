import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    def _arrange(input):
        arranged = input.flatten(end_dim=-1)
        arranged = arranged.tile((block_size, 1))
        arranged = arranged.tile((1, -1))
        arranged.dtype = arranged.dtype.squeeze(0)

        return arranged

    return _arrange(input), _arrange(output)


def application(input, output):
    denominators = ntl.sqrt(input[0] * input[0] + input[1] * input[1])
    denominators = ntl.where(denominators == 0.0, 1.0, denominators)

    for i in range(input.shape[0]):
        output[i] = input[i] / denominators  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
