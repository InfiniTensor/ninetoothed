import torch

import ntops
from ntops.torch.utils import _cached_make


def gelu(input, approximate="none"):
    output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.gelu.premake, input.ndim, approximate)

    kernel(input, output)

    return output
