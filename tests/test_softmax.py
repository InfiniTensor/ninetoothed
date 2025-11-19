import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor
from tests.utils import get_available_devices


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


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("n", (781,))
@pytest.mark.parametrize("m", (1823,))
def test(m, n, dtype, device):
    torch.manual_seed(0)

    input = torch.rand((m, n), dtype=dtype, device=device)

    output = softmax(input)
    expected = torch.softmax(input, dim=-1)

    assert torch.allclose(output, expected)
