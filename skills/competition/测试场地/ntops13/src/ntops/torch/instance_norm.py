import math

import torch

import ntops
from ntops.torch.utils import _cached_make


def instance_norm(
    input,
    running_mean=None,
    running_var=None,
    weight=None,
    bias=None,
    use_input_stats=True,
    momentum=0.1,
    eps=1e-05,
):
    if weight is None:
        weight = torch.ones(input.shape[1], device=input.device, dtype=input.dtype)

    if bias is None:
        bias = torch.zeros(input.shape[1], device=input.device, dtype=input.dtype)

    has_running_stats = running_mean is not None and running_var is not None

    if use_input_stats:
        mean = torch.empty(input.shape[:2], device=input.device, dtype=input.dtype)
        var = torch.empty(input.shape[:2], device=input.device, dtype=input.dtype)

    output = torch.empty_like(input)

    num_normalized_elements = math.prod(input.shape[2:])
    kernel = _cached_make(
        ntops.kernels.instance_norm.premake,
        input.ndim,
        use_input_stats,
        num_normalized_elements,
        dtype=input.dtype,
    )

    if use_input_stats:
        kernel(
            input,
            mean,
            var,
            weight,
            bias,
            eps,
            output,
            num_normalized_elements,
        )

        # We reduce in PyTorch instead of using tl.atomic_add in Triton because:
        # 1. Triton blocks cannot synchronize to safely apply the momentum update after all additions finish.
        # 2. N blocks atomically adding to the same C addresses creates severe memory contention.
        if has_running_stats:
            batch_mean = mean.mean(0)
            avg_vars = var.mean(0)

            unbiased_var = (
                (avg_vars) * num_normalized_elements / (num_normalized_elements - 1)
                if num_normalized_elements > 1
                else avg_vars
            )

            running_mean.mul_(1 - momentum).add_(momentum * batch_mean)
            running_var.mul_(1 - momentum).add_(momentum * unbiased_var)
    else:
        kernel(input, running_mean, running_var, weight, bias, eps, output)

    return output
