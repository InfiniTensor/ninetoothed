"""T2 self-test — L2 reduction / block lane.

Operator: row-wise softmax over the last dimension.

Follows `tests/test_softmax.py`: tile a row as `(1, BLOCK_SIZE)`, use a
`constexpr` block size set to the row width at call time, and reduce with
`ntl.max`, `ntl.exp`, `ntl.sum`. The input is declared with `other=float("-inf")`
so out-of-bounds lanes do not affect the max/sum reduction — the key boundary
rule for the reduction lane.

Run from the repository root:
    pytest skills/competition/ninetoothed-operator-skill/examples/t2-reduction-block/test_softmax_op.py -q -p no:cacheprovider
"""

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
    input = torch.rand((m, n), dtype=dtype, device=device)

    output = softmax(input)
    expected = torch.softmax(input, dim=-1)

    assert torch.allclose(output, expected)
