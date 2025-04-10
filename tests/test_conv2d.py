import functools

import torch
import torch.nn.functional as F

import ninetoothed
import tests.test_matmul as matmul
from ninetoothed import Tensor
from tests.skippers import skip_if_cuda_not_available


def arrangement(
    input,
    filter,
    output,
    BLOCK_SIZE_M=matmul.BLOCK_SIZE_M,
    BLOCK_SIZE_N=matmul.BLOCK_SIZE_N,
    BLOCK_SIZE_K=matmul.BLOCK_SIZE_K,
):
    input_tiled = input.tile((1, *filter.shape[1:]), strides=(-1, -1, 1, 1))
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


def conv2d(input, filter):
    n, _, h, w = input.shape
    k, _, r, s = filter.shape
    p = h - r + 1
    q = w - s + 1

    output = torch.empty((n, k, p, q), device=input.device, dtype=input.dtype)

    conv2d_kernel = ninetoothed.make(
        arrangement,
        matmul.application,
        (Tensor(4), Tensor(4, shape_options={"constexpr": True}), Tensor(4)),
        max_num_configs=50,
    )

    conv2d_kernel(input, filter, output)

    return output


@skip_if_cuda_not_available
class TestCUDA:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)

        n, c, h, w = 4, 64, 16, 16
        k, _, r, s = 512, c, 3, 3

        cls.input = torch.randn(n, c, h, w, device="cuda")
        cls.filter = torch.randn(k, c, r, s, device="cuda")

    def test_fp16(self):
        input = type(self).input.to(torch.float16)
        filter = type(self).filter.to(torch.float16)

        assert torch.allclose(
            conv2d(input, filter), F.conv2d(input, filter), atol=0.001, rtol=0.001
        )
