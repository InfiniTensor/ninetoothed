import ninetoothed


def arrangement(*tensors, dim, block_size=None):
    dims = dim

    if isinstance(dims, int):
        dims = (dims,)

    if block_size is None:
        block_size = ninetoothed.block_size()

    ndim = max(tensor.ndim for tensor in tensors)

    assert all(tensor.ndim == ndim or tensor.ndim == 0 for tensor in tensors)

    dims = tuple(dim if dim >= 0 else dim + ndim for dim in dims)

    non_target_dims = tuple(i for i in range(ndim) if i not in dims)

    def _arrange(tensor):
        arranged = tensor.permute(non_target_dims + dims)
        arranged = arranged.flatten(start_dim=-len(dims))

        inner_block_shape = tuple(1 for _ in non_target_dims) + (block_size,)
        outer_block_shape = tuple(1 for _ in non_target_dims) + (-1,)
        non_target_dim_indices = tuple(range(len(non_target_dims)))

        arranged = arranged.tile(inner_block_shape)
        arranged = arranged.tile(outer_block_shape)
        arranged.dtype = arranged.dtype.squeeze(non_target_dim_indices)
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze(non_target_dim_indices)

        return arranged

    return tuple(_arrange(tensor) if tensor.ndim != 0 else tensor for tensor in tensors)
