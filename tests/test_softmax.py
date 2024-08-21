import torch
import triton

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from tests.skippers import skip_if_cuda_not_available


def softmax(input):
    output = torch.empty_like(input)

    block_size = triton.next_power_of_2(input.shape[-1])

    @ninetoothed.jit
    def softmax_kernel(
        input_row: Tensor(2, other=float("-inf")).tile((1, block_size)),
        output_row: Tensor(2).tile((1, block_size)),
    ):
        row_minus_max = input_row - ntl.max(input_row)
        numerator = ntl.exp(row_minus_max)
        denominator = ntl.sum(numerator)
        output_row = numerator / denominator  # noqa: F841

    softmax_kernel(input, output)

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
