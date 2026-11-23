import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("device", ("cuda",))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize("ceil_mode", (False,))
@pytest.mark.parametrize("dilation", (1, 2, (2, 3)))
@pytest.mark.parametrize("padding", (0, 1, (2, 3)))
@pytest.mark.parametrize("stride", (None, 1, (2, 3)))
@pytest.mark.parametrize("kernel_size", ((1, 1), (3, 3)))
@pytest.mark.parametrize("n, c, h, w", ((2, 3, 112, 112),))
def test_max_pool2d(
    n, c, h, w, kernel_size, stride, padding, dilation, ceil_mode, dtype, device
):
    padding_ = padding

    if isinstance(padding_, int):
        padding_ = (padding_, padding_)

    if padding_[0] > kernel_size[0] / 2 or padding_[1] > kernel_size[1] / 2:
        pytest.skip(reason="Invalid padding.")

    input = torch.randn((n, c, h, w), dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.max_pool2d(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    reference_output = F.max_pool2d(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    assert torch.allclose(ninetoothed_output, reference_output)
