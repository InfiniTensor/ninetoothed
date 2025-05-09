import random

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size
from tests.skippers import skip_if_cuda_not_available


def arrangement(input, p, seed, output, BLOCK_SIZE=block_size()):
    return input.tile((BLOCK_SIZE,)), p, seed, output.tile((BLOCK_SIZE,))


def application(input, p, seed, output):
    output = ntl.where(ntl.rand(seed, input.offsets()) > p, input / (1 - p), 0)  # noqa: F841


def dropout(input, p=0.5):
    seed = random.randrange(0, 2**31)
    output = torch.empty_like(input)

    tensors = (Tensor(1), Tensor(0), Tensor(0), Tensor(1))
    dropout_kernel = ninetoothed.make(arrangement, application, tensors)

    dropout_kernel(input, p, seed, output)

    return output


@skip_if_cuda_not_available
class TestCUDA:
    @classmethod
    def setup_class(cls):
        random.seed(0)
        torch.manual_seed(0)

        size = 349

        cls.input = torch.randn(size, device="cuda")

    def test_fp16(self):
        input = type(self).input.to(torch.float16)
        p = 0.3

        output = dropout(input, p=p)

        assert input.shape == output.shape

        non_zero_ratio = output.nonzero().numel() / input.numel()

        assert abs(non_zero_ratio - (1 - p)) < 0.05

        assert torch.allclose(output[output != 0], input[output != 0] / (1 - p))
