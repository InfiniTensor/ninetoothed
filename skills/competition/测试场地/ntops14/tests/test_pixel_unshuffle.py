import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("downscale_factor", (2, 3, 4))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_pixel_unshuffle_4d(dtype, downscale_factor):
    r = downscale_factor
    N, C, H, W = 2, 3, 8, 8
    input = torch.randn(N, C, H * r, W * r, dtype=dtype, device="cuda")

    reference = F.pixel_unshuffle(input, r)
    result = ntops.torch.pixel_unshuffle(input, r)

    assert result.shape == reference.shape
    assert torch.allclose(result, reference, rtol=1e-3, atol=1e-3)
    assert result.is_contiguous()


@skip_if_cuda_not_available
@pytest.mark.parametrize("downscale_factor", (2, 3))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_pixel_un_shuffle_3d(dtype, downscale_factor):
    r = downscale_factor
    C, H, W = 4, 6, 6
    input = torch.randn(C, H * r, W * r, dtype=dtype, device="cuda")

    reference = F.pixel_unshuffle(input, r)
    result = ntops.torch.pixel_unshuffle(input, r)

    assert result.shape == reference.shape
    assert torch.allclose(result, reference, rtol=1e-3, atol=1e-3)
    assert result.is_contiguous()


@skip_if_cuda_not_available
def test_pixel_un_shuffle_invalid_shape():
    input = torch.randn(2, 3, 7, 8, device="cuda")
    with pytest.raises(AssertionError):
        ntops.torch.pixel_unshuffle(input, 3)


@skip_if_cuda_not_available
def test_pixel_un_shuffle_invalid_ndim():
    input = torch.randn(2, 3, 4, 8, 8, device="cuda")
    with pytest.raises(ValueError):
        ntops.torch.pixel_unshuffle(input, 2)
