import math

import torch

import ntops
import ntops.kernels.nansum
from ntops.torch.utils import _cached_make


def _next_power_of_2(n):
    if n == 0:
        return 1
    return 1 << (n - 1).bit_length()


def _get_optimal_block_size(dim_size):
    target = _next_power_of_2(dim_size)
    return max(32, min(target, 1024))


def nansum(input, dim=None, keepdim=False, *, dtype=None):
    computation_dtype = dtype if dtype is not None else input.dtype

    # --- Case A: Global nansum (dim=None) ---
    if dim is None:
        current = input
        block_size = _get_optimal_block_size(current.numel())

        while current.numel() > 1:
            output_len = math.ceil(current.numel() / block_size)
            output = torch.empty(
                (output_len,), dtype=computation_dtype, device=current.device
            )
            kernel = _cached_make(
                ntops.kernels.nansum.premake_all_elements,
                current.ndim,
                computation_dtype,
                block_size,
            )
            kernel(current, output)
            current = output

        result = current.view(())

        if keepdim:
            result = result.reshape([1] * input.ndim)

        return result

    # --- Case B: Dim nansum ---
    if isinstance(dim, int):
        dims = (dim,)
    else:
        dims = tuple(dim)

    dims = tuple(d if d >= 0 else d + input.ndim for d in dims)

    output_shape = list(input.shape)
    for d in dims:
        output_shape[d] = 1

    dim_size = 1
    for d in dims:
        dim_size *= input.shape[d]
    block_size = _get_optimal_block_size(dim_size)

    temp_out = torch.empty(output_shape, dtype=computation_dtype, device=input.device)

    kernel = _cached_make(
        ntops.kernels.nansum.premake,
        input.ndim,
        dims,
        computation_dtype,
        block_size,
    )
    kernel(input, temp_out)

    if not keepdim:
        dims_to_remove = sorted(dims, reverse=True)
        final_shape = list(output_shape)
        for d in dims_to_remove:
            del final_shape[d]
        if not final_shape:
            temp_out = temp_out.view(())
        else:
            temp_out = temp_out.view(final_shape)

    return temp_out
