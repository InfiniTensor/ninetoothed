import functools

import pytest
import torch
import torch.nn.functional as F

import ninetoothed
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

    input_tiled = input_padded.tile((1, *filter.shape[1:]), strides=(-1, -1, 1, 1))
    input_squeezed = input_tiled.squeeze(1)
    input_squeezed.dtype = input_squeezed.dtype.squeeze(0)
    input_raveled = input_squeezed.ravel()
    input_flattened = input_raveled.flatten(end_dim=3).flatten(start_dim=1)

    filter_flattened = filter.flatten(start_dim=1)
    filter_permuted = filter_flattened.permute((1, 0))

    output_flattened = output.permute((0, 2, 3, 1)).flatten(end_dim=3)

    return functools.partial(
        matmul.arrangement,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )(input_flattened, filter_permuted, output_flattened)


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
        max_num_configs=50,
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
