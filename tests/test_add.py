import pytest
import torch

import ninetoothed
from ninetoothed import Symbol, Tensor
from tests.utils import get_available_devices


def add(lhs, rhs):
    BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)

    @ninetoothed.jit
    def add_kernel(
        lhs: Tensor(1).tile((BLOCK_SIZE,)),
        rhs: Tensor(1).tile((BLOCK_SIZE,)),
        output: Tensor(1).tile((BLOCK_SIZE,)),
    ):
        output = lhs + rhs  # noqa: F841

    output = torch.empty_like(lhs)

    add_kernel(lhs, rhs, output)

    return output


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("size", (98432,))
def test(size, dtype, device):
    torch.manual_seed(0)

    input = torch.rand(size, dtype=dtype, device=device)
    other = torch.rand(size, dtype=dtype, device=device)

    output = add(input, other)
    expected = input + other

    assert torch.allclose(output, expected)
