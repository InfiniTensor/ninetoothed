import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor
from tests.skippers import skip_if_cuda_not_available


def softmax(input):
    BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)

    @ninetoothed.jit
    def softmax_kernel(
        input_row: Tensor(2, other=float("-inf")).tile((1, BLOCK_SIZE)),
        output_row: Tensor(2).tile((1, BLOCK_SIZE)),
    ):
        row_minus_max = input_row - ntl.max(input_row)
        numerator = ntl.exp(row_minus_max)
        denominator = ntl.sum(numerator)
        output_row = numerator / denominator  # noqa: F841

    output = torch.empty_like(input)

    softmax_kernel(input, output, BLOCK_SIZE=input.shape[-1])

    return output


@skip_if_cuda_not_available
class TestCUDA:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)

        cls.input = torch.randn(1823, 781, device="cuda")

    def test_fp32(self):
        input = type(self).input.to(torch.float32)

        assert torch.allclose(softmax(input), torch.softmax(input, axis=-1))
