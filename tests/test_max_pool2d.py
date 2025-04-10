import functools

import torch
import torch.nn.functional as F

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor
from tests.skippers import skip_if_cuda_not_available


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


@skip_if_cuda_not_available
class TestCUDA:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)

        cls.input = torch.randn(32, 3, 64, 64, device="cuda")
        cls.window_shape = (3, 3)

    def test_fp16_ceil(self):
        input = type(self).input.to(torch.float16)
        window_shape = type(self).window_shape

        assert torch.allclose(
            max_pool2d(input, window_shape, ceil_mode=True),
            F.max_pool2d(input, window_shape, ceil_mode=True),
        )

    def test_fp16_floor(self):
        input = type(self).input.to(torch.float16)
        window_shape = type(self).window_shape

        assert torch.allclose(
            max_pool2d(input, window_shape, ceil_mode=False),
            F.max_pool2d(input, window_shape, ceil_mode=False),
        )
