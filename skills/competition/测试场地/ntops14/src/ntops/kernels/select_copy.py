import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, index, output, dim, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    if output.ndim < 1:
        output = output.unsqueeze(0)
    else:
        output = output.flatten()

    output_arranged = output.tile((1,))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    if input.ndim < 2:
        input = input.unsqueeze(0)
    else:
        if dim < 0:
            dim += input.ndim

        non_target_dims = tuple(i for i in range(input.ndim) if i != dim)
        input = input.permute(non_target_dims + (dim,))

    input_arranged = input.flatten(end_dim=-1)
    input_arranged = input_arranged.tile((1, -1))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    return input_arranged, index, output_arranged


def application(input, index, output):
    idx = ntl.cast(index, ntl.int32)
    output = input[idx]  # noqa: F841


def premake(in_ndim, out_ndim, dim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    tensors = (
        Tensor(in_ndim, dtype=dtype, shape_options={"constexpr": True}),
        Tensor(0, dtype=ninetoothed.int32),
        Tensor(out_ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
