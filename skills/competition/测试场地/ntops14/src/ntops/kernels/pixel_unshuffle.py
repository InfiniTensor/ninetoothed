import functools

import ninetoothed
from ninetoothed import Tensor


def arrangement(source, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    return (
        source.flatten().tile((block_size,)),
        output.flatten().tile((block_size,)),
    )


def application(source, output):
    output = source  # noqa: F841


def premake(source_ndim, output_ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(source_ndim, dtype=dtype),
        Tensor(output_ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
