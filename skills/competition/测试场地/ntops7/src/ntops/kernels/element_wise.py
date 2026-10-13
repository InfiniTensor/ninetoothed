import ninetoothed


def arrangement(*tensors, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    ndim = max(tensor.ndim for tensor in tensors)

    assert all(tensor.ndim == ndim or tensor.ndim == 0 for tensor in tensors)

    return tuple(
        tensor.flatten().tile((block_size,)) if tensor.ndim != 0 else tensor
        for tensor in tensors
    )
