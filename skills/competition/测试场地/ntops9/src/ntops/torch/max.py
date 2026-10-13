import torch

import ntops
import ninetoothed
from ntops.torch.utils import _cached_make
import builtins
import math

def max(input, dim: int | None = None, keepdim=False, *, out=None):
    if dim is None:
        current = input

        # 递归地应用 max kernel 直到只剩一个元素
        block_size = 1024
        while current.numel() > 1:
            output_shape = (math.ceil(current.numel() / block_size),)
            output = torch.empty(output_shape, dtype=current.dtype, device=current.device)
            kernel = _cached_make(ntops.kernels.max.premake_all_elements, current.ndim, current.dtype, block_size)
            kernel(current, output)
            current = output
        
        result = current.view(())
        
        if out is not None:
            out.copy_(result)
            return out
        
        return result
    else:
        output_shape = list(input.shape)
        output_shape[dim] = 1

        temp_out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        temp_out_idx = torch.empty(output_shape, dtype=torch.int64, device=input.device)

        block_size = 1024
        kernel = _cached_make(ntops.kernels.max.premake, input.ndim, dim, block_size)
        kernel(input, temp_out, temp_out_idx)

        if not keepdim:
            del output_shape[dim]
            temp_out = temp_out.view(output_shape)
            temp_out_idx = temp_out_idx.view(output_shape)

        if out is not None:
            out.copy_(temp_out)
            return out, temp_out_idx

        return temp_out, temp_out_idx
