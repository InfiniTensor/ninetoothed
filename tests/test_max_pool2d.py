import math

import torch
import torch.nn.functional as F

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor
from tests.skippers import skip_if_cuda_not_available


def arrangement(input, output):
    BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)

    WINDOW_HEIGHT = Symbol("WINDOW_HEIGHT", constexpr=True)
    WINDOW_WIDTH = Symbol("WINDOW_WIDTH", constexpr=True)

    input_arranged = input.tile((1, 1, WINDOW_HEIGHT, WINDOW_WIDTH))
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


def max_pool2d(input, window_shape):
    n, c, h, w = input.shape
    r, s = window_shape
    p = math.ceil((h - r) / r + 1)
    q = math.ceil((w - s) / s + 1)

    output = torch.empty(n, c, p, q, dtype=input.dtype, device=input.device)

    max_pool2d_kernel = ninetoothed.make(
        arrangement, application, (Tensor(4, other=float("-inf")), Tensor(4))
    )

    max_pool2d_kernel(input, output, WINDOW_HEIGHT=r, WINDOW_WIDTH=s)

    return output


@skip_if_cuda_not_available
class TestCUDA:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)

        cls.input = torch.randn(32, 3, 64, 64, device="cuda")
        cls.window_shape = (3, 3)

    def test_fp16(self):
        input = type(self).input.to(torch.float16)
        window_shape = type(self).window_shape

        assert torch.allclose(
            max_pool2d(input, window_shape),
            F.max_pool2d(input, window_shape, ceil_mode=True),
        )
