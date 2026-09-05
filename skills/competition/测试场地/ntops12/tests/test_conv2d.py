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
@pytest.mark.parametrize("dilation", (1, 2, (2, 3)))
@pytest.mark.parametrize("padding", (0, 1, (2, 3)))
@pytest.mark.parametrize("stride", (1, 2, (2, 3)))
@pytest.mark.parametrize("r, s", ((1, 1), (3, 3)))
@pytest.mark.parametrize("n, c, h, w, k", ((2, 3, 112, 112, 4),))
def test_conv2d(
    n, c, h, w, k, r, s, stride, padding, dilation, dtype, device, rtol, atol
):
    input = torch.randn((n, c, h, w), dtype=dtype, device=device)
    weight = torch.randn((k, c, r, s), dtype=dtype, device=device)
    bias = torch.randn((k,), dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.conv2d(
        input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation
    )
    reference_output = F.conv2d(
        input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation
    )

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
