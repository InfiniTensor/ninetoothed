import math
import torch
import ntops
from ntops.torch.utils import _cached_make

def batch_norm(input, weight=None, bias=None, eps=1e-5, training=True):
    ndim = input.ndim
    if ndim < 2:
        raise ValueError("Input to batch_norm must have at least 2 dimensions")

    # 假设 dim=1 是 Channel，其余为 Batch 和 Spatial
    channel_dim = 1
    reduction_dims = tuple(d for d in range(ndim) if d != channel_dim)
    
    num_elements = 1
    for d in reduction_dims:
        num_elements *= input.shape[d]

    # 构造 Broadcasting 形状: (1, C, 1, 1...)
    C = input.shape[channel_dim]
    shape_for_broadcast = [1] * ndim
    shape_for_broadcast[channel_dim] = C

    if weight is None:
        weight = torch.ones(C, dtype=input.dtype, device=input.device)
    if bias is None:
        bias = torch.zeros(C, dtype=input.dtype, device=input.device)

    # 扩展到全尺寸，交给 Kernel 处理
    weight_expanded = weight.view(*shape_for_broadcast).expand_as(input)
    bias_expanded = bias.view(*shape_for_broadcast).expand_as(input)

    output = torch.empty_like(input)

    kernel = _cached_make(
        ntops.kernels.batch_norm.premake, 
        ndim, 
        reduction_dims, 
        num_elements
    )

    kernel(input, weight_expanded, bias_expanded, eps, output, num_elements)

    return output