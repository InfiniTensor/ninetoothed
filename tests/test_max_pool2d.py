import functools

import pytest
import torch
import torch.nn.functional as F

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor
from tests.utils import get_available_devices


def arrangement(input, output, ceil_mode=False):
    BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)

    WINDOW_HEIGHT = Symbol("WINDOW_HEIGHT", constexpr=True, upper_bound=16)
    WINDOW_WIDTH = Symbol("WINDOW_WIDTH", constexpr=True, upper_bound=16)

    input_arranged = input.tile(
        (1, 1, WINDOW_HEIGHT, WINDOW_WIDTH), floor_mode=not ceil_mode
    )
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=4).flatten(start_dim=1)
    input_arranged = input_arranged.tile((BLOCK_SIZE, -1))

    output_arranged = output.tile((1, 1, 1, 1))
    output_arranged = output_arranged.ravel()
    output_arranged = output_arranged.flatten(end_dim=4).flatten(start_dim=1)
    output_arranged = output_arranged.tile((BLOCK_SIZE, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)

    return input_arranged, output_arranged


def application(input, output):
    output = ntl.max(input, axis=1)  # noqa: F841


def max_pool2d(input, window_shape, ceil_mode=False):
    def _div(x, y, ceil_mode=False):
        if ceil_mode:
            return (x + y - 1) // y

        return x // y

    n, c, h, w = input.shape
    r, s = window_shape
    p = _div(h - r, r, ceil_mode=ceil_mode) + 1
    q = _div(w - s, s, ceil_mode=ceil_mode) + 1

    output = torch.empty(n, c, p, q, dtype=input.dtype, device=input.device)

    max_pool2d_kernels = {
        ceil_mode: ninetoothed.make(
            functools.partial(arrangement, ceil_mode=ceil_mode),
            application,
            (Tensor(4, other=float("-inf")), Tensor(4)),
        )
        for ceil_mode in (True, False)
    }

    max_pool2d_kernels[ceil_mode](input, output, WINDOW_HEIGHT=r, WINDOW_WIDTH=s)

    return output


@pytest.mark.parametrize("ceil_mode", (False, True))
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float16,))
@pytest.mark.parametrize("s", (3,))
@pytest.mark.parametrize("r", (3,))
@pytest.mark.parametrize("w", (64,))
@pytest.mark.parametrize("h", (64,))
@pytest.mark.parametrize("c", (3,))
@pytest.mark.parametrize("n", (32,))
def test(n, c, h, w, r, s, dtype, device, ceil_mode):
    torch.manual_seed(0)

    input = torch.randn((n, c, h, w), dtype=dtype, device=device)
    window_shape = (r, s)

    output = max_pool2d(input, window_shape, ceil_mode=ceil_mode)
    expected = F.max_pool2d(input, window_shape, ceil_mode=ceil_mode)

    assert torch.allclose(output, expected)
