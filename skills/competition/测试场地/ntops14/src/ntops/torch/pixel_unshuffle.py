import torch

import ntops
from ntops.torch.utils import _cached_make


def pixel_unshuffle(input, downscale_factor, *, out=None):
    r = downscale_factor

    if input.ndim == 3:
        C, H_r, W_r = input.shape
        assert H_r % r == 0, f"H ({H_r}) must be divisible by downscale_factor ({r})"
        assert W_r % r == 0, f"W ({W_r}) must be divisible by downscale_factor ({r})"
        H = H_r // r
        W = W_r // r

        if out is None:
            out = torch.empty((C * r * r, H, W), dtype=input.dtype, device=input.device)

        intermediate = input.view(C, H, r, W, r).permute(0, 2, 4, 1, 3)
        source_ndim = intermediate.ndim
        output_ndim = out.ndim
    elif input.ndim == 4:
        N, C, H_r, W_r = input.shape
        assert H_r % r == 0, f"H ({H_r}) must be divisible by downscale_factor ({r})"
        assert W_r % r == 0, f"W ({W_r}) must be divisible by downscale_factor ({r})"
        H = H_r // r
        W = W_r // r

        if out is None:
            out = torch.empty(
                (N, C * r * r, H, W), dtype=input.dtype, device=input.device
            )

        intermediate = input.view(N, C, H, r, W, r).permute(0, 1, 3, 5, 2, 4)
        source_ndim = intermediate.ndim
        output_ndim = out.ndim
    else:
        raise ValueError(
            f"Expected 3D or 4D input, got {input.ndim}D"
        )

    kernel = _cached_make(
        ntops.kernels.pixel_unshuffle.premake,
        source_ndim,
        output_ndim,
        input.dtype,
        4096,
    )

    kernel(intermediate, out)

    return out
