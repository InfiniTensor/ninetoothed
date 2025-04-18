import torch

import ninetoothed
from ninetoothed import Tensor, block_size
from ninetoothed.language import libdevice
from tests.skippers import skip_if_cuda_not_available


def arrangement(input, exponent, output, BLOCK_SIZE=block_size()):
    return (
        input.tile((BLOCK_SIZE,)),
        exponent.tile((BLOCK_SIZE,)),
        output.tile((BLOCK_SIZE,)),
    )


def application(input, exponent, output):
    output = libdevice.pow(input, exponent)  # noqa: F841


def pow(input, exponent):
    output = torch.empty_like(input)

    pow_kernel = ninetoothed.make(
        arrangement, application, (Tensor(1), Tensor(1), Tensor(1))
    )

    pow_kernel(input, exponent, output)

    return output


@skip_if_cuda_not_available
class TestCUDA:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)

        size = 44925

        cls.input = torch.randn(size, device="cuda")
        cls.exponent = torch.randn(size, device="cuda")

    def test_fp32(self):
        input = type(self).input.to(torch.float32)
        exponent = type(self).exponent.to(torch.float32)

        assert torch.allclose(
            pow(input, exponent), torch.pow(input, exponent), equal_nan=True
        )
