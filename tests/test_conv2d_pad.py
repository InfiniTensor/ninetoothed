import functools

import pytest
import torch
import torch.nn.functional as F

import ninetoothed
import tests.test_conv2d as conv2d
import tests.test_matmul as matmul
from ninetoothed import Tensor
from tests.utils import get_available_devices


def arrangement(
    input,
    filter,
    output,
    padding=(0, 0),
    BLOCK_SIZE_M=matmul.BLOCK_SIZE_M,
    BLOCK_SIZE_N=matmul.BLOCK_SIZE_N,
    BLOCK_SIZE_K=matmul.BLOCK_SIZE_K,
):
    pad_h, pad_w = padding
    input_padded = input.pad((pad_w, pad_w, pad_h, pad_h))

    return conv2d.arrangement(
        input_padded, filter, output, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )


def conv2d_pad(input, filter, padding=(0, 0)):
    n, _, h, w = input.shape
    k, _, r, s = filter.shape
    pad_h, pad_w = padding
    p = h + 2 * pad_h - r + 1
    q = w + 2 * pad_w - s + 1

    output = torch.empty((n, k, p, q), device=input.device, dtype=input.dtype)

    conv2d_kernel = ninetoothed.make(
        functools.partial(arrangement, padding=padding),
        matmul.application,
        (Tensor(4), Tensor(4, shape_options={"constexpr": True}), Tensor(4)),
        max_num_configs=2,
    )

    conv2d_kernel(input, filter, output)

    return output


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("padding", ((1, 1), (0, 1), (2, 0)))
@pytest.mark.parametrize("s", (3,))
@pytest.mark.parametrize("r", (3,))
@pytest.mark.parametrize("k", (64,))
@pytest.mark.parametrize("w", (16,))
@pytest.mark.parametrize("h", (16,))
@pytest.mark.parametrize("c", (32,))
@pytest.mark.parametrize("n", (2,))
def test_conv2d_pad(n, c, h, w, k, r, s, padding, dtype, device):
    rtol, atol = 1e-1, 1e-1

    input = torch.randn((n, c, h, w), dtype=dtype, device=device)
    weight = torch.randn((k, c, r, s), dtype=dtype, device=device)

    output = conv2d_pad(input, weight, padding=padding)
    expected = F.conv2d(input, weight, padding=padding)

    diff = (output - expected).abs().max().item()
    assert torch.allclose(output, expected, rtol=rtol, atol=atol), f"Max diff: {diff}"
