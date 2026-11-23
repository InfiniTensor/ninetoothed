import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("device", ("cuda",))
@pytest.mark.parametrize(
    "dtype, rtol, atol", ((torch.float32, 1e-5, 1e-5), (torch.float16, 1e-3, 1e-3))
)
@pytest.mark.parametrize("ceil_mode", (False,))
@pytest.mark.parametrize("padding", (0, 1, (2, 3)))
@pytest.mark.parametrize("stride", (None, 1, (2, 3)))
@pytest.mark.parametrize("kernel_size", ((1, 1), (3, 3)))
@pytest.mark.parametrize("n, c, h, w", ((2, 3, 112, 112),))
def test_avg_pool2d(
    n, c, h, w, kernel_size, stride, padding, ceil_mode, dtype, device, rtol, atol
):
    padding_ = padding

    if isinstance(padding_, int):
        padding_ = (padding_, padding_)

    if padding_[0] > kernel_size[0] / 2 or padding_[1] > kernel_size[1] / 2:
        pytest.skip(reason="Invalid padding.")

    input = torch.randn((n, c, h, w), dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.avg_pool2d(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
    )
    reference_output = F.avg_pool2d(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
    )

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
