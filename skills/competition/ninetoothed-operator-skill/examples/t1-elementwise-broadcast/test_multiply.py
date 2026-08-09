"""T1 self-test — L1 elementwise / broadcast lane.

Operator: elementwise multiply (Hadamard product), output = lhs * rhs.

Chosen to differ from the repo's `add` example (proves the skill generalizes
beyond the sample) while using only the `*` primitive, which is as safe as the
`+` used in `tests/test_add.py`. Follows the decorator form from that test.

Run from the repository root:
    pytest skills/competition/ninetoothed-operator-skill/examples/t1-elementwise-broadcast/test_multiply.py -q -p no:cacheprovider
"""

import pytest
import torch

import ninetoothed
from ninetoothed import Symbol, Tensor
from tests.utils import get_available_devices


def multiply(lhs, rhs):
    BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)

    @ninetoothed.jit
    def multiply_kernel(
        lhs: Tensor(1).tile((BLOCK_SIZE,)),
        rhs: Tensor(1).tile((BLOCK_SIZE,)),
        output: Tensor(1).tile((BLOCK_SIZE,)),
    ):
        output = lhs * rhs  # noqa: F841

    output = torch.empty_like(lhs)

    multiply_kernel(lhs, rhs, output)

    return output


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("size", (98432,))
def test(size, dtype, device):
    lhs = torch.rand(size, dtype=dtype, device=device)
    rhs = torch.rand(size, dtype=dtype, device=device)

    output = multiply(lhs, rhs)
    expected = lhs * rhs

    assert torch.allclose(output, expected)
