import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, output, k, dims, dim_sizes, block_size=None):
    def _arrange(input, output, dim, dim_size):
        dims = dim
        dim_sizes = dim_size

        if isinstance(dims, int):
            dims = (dims,)

        if isinstance(dim_sizes, int):
            dim_sizes = (dim_sizes,)

        assert len(dims) == len(dim_sizes)

        ndim = input.ndim
        dims = tuple(dim if dim >= 0 else dim + ndim for dim in dims)
        non_target_dims = tuple(i for i in range(ndim) if i not in dims)

        input_arranged = input.permute(non_target_dims + dims)
        output_arranged = output.permute(non_target_dims + dims)

        input_arranged = input_arranged.pad(
            tuple((0, 0) for _ in non_target_dims)
            + tuple(((-size) % block_size, 0) for size in dim_sizes)
        )

        inner_block_shape = tuple(1 for _ in non_target_dims) + tuple(
            block_size for _ in range(len(dims))
        )
        outer_block_shape = tuple(1 for _ in non_target_dims) + tuple(
            -1 for _ in range(len(dims))
        )
        non_target_dim_indices = tuple(range(len(non_target_dims)))

        arranged = []
        for tensor in (input_arranged, output_arranged):
            tensor = tensor.tile(inner_block_shape)
            tensor = tensor.tile(outer_block_shape)
            tensor.dtype = tensor.dtype.squeeze(non_target_dim_indices)
            tensor.dtype.dtype = tensor.dtype.dtype.squeeze(non_target_dim_indices)
            arranged.append(tensor)

        return tuple(arranged)

    def _transpose(tensor, dims):
        perm = list(range(tensor.ndim))
        for i, dim in enumerate(dims):
            perm[dim] = dims[(i + 1) % len(dims)]

        return tensor.permute(perm)

    if block_size is None:
        # `block_size` is used to compute paddings, so it cannot be `ninetoothed.Symbol`.
        block_size = 32

    if k % 4 == 0:
        input_arranged = input.flatten().tile((block_size,))
        output_arranged = output.flatten().tile((block_size,))
    else:
        if k % 2 == 1:
            input = _transpose(input, dims)
            dims = dims[(k % 4) >> 1]
            dim_sizes = dim_sizes[((k % 4) >> 1) ^ 1]

        input_arranged, output_arranged = _arrange(input, output, dims, dim_sizes)

    return input_arranged, output_arranged


def application_copy(input, output):
    output = input  # noqa: F841


def application_1D_flip(input, output):
    i_size = input.shape[0]
    for i in range(i_size):
        output[i] = ntl.flip(input[i_size - 1 - i], 0)


def application_2D_flip(input, output):
    i_size = input.shape[0]
    j_size = input.shape[1]
    for i in range(i_size):
        for j in range(j_size):
            output[i, j] = ntl.flip(
                ntl.flip(input[i_size - 1 - i, j_size - 1 - j], 0), 1
            )


def premake(ndim, k, dims, dim_sizes, dtype=None, block_size=None):
    arrangement_ = functools.partial(
        arrangement, k=k, dims=dims, dim_sizes=dim_sizes, block_size=block_size
    )

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    if k % 4 == 0:
        application = application_copy
    elif k % 4 == 2:
        application = application_2D_flip
    else:
        application = application_1D_flip

    return arrangement_, application, tensors
