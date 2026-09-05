import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    ndim = input.ndim
    dim = 0

    non_target_dims = tuple(i for i in range(input.ndim) if i != dim)

    def _arrangement(input):
        arranged = input.permute(non_target_dims + (dim,))

        if ndim == 1:
            arranged = arranged.unsqueeze(0)
        arranged = arranged.flatten(end_dim=-1)

        arranged = arranged.tile((1, -1))
        arranged.dtype = arranged.dtype.squeeze(0)

        return arranged

    return _arrangement(input), _arrangement(output)


def application(input, output):
    output = ntl.sort(input)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(
            ndim, dtype=dtype, other=float("inf"), shape_options={"constexpr": True}
        ),
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),
    )

    return arrangement_, application, tensors
