"""T3 self-test — L3 layout-sensitive lane.

Operator: 2-D elementwise multiply, exercised on BOTH contiguous and
non-contiguous (transposed) inputs.

The defining risk of the layout lane is silently assuming contiguous inputs.
This test deliberately feeds a transposed tensor (`x.T`, which is non-contiguous
with swapped strides) and checks the result still matches the PyTorch reference
computed on the same non-contiguous tensor. If NineToothed mishandled strides,
the contiguous case would pass while the transposed case would fail.

Run from the repository root:
    pytest skills/competition/ninetoothed-operator-skill/examples/t3-layout-sensitive/test_multiply_2d.py -q -p no:cacheprovider
"""

import pytest
import torch

import ninetoothed
from ninetoothed import Symbol, Tensor
from tests.utils import get_available_devices

BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)


def multiply_2d(lhs, rhs):
    @ninetoothed.jit
    def multiply_2d_kernel(
        lhs: Tensor(2).tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
        rhs: Tensor(2).tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
        output: Tensor(2).tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
    ):
        output = lhs * rhs  # noqa: F841

    output = torch.empty_like(lhs)

    multiply_2d_kernel(lhs, rhs, output)

    return output


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("n", (512,))
@pytest.mark.parametrize("m", (384,))
def test_contiguous(m, n, dtype, device):
    lhs = torch.rand((m, n), dtype=dtype, device=device)
    rhs = torch.rand((m, n), dtype=dtype, device=device)

    output = multiply_2d(lhs, rhs)
    expected = lhs * rhs

    assert torch.allclose(output, expected)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("n", (512,))
@pytest.mark.parametrize("m", (384,))
def test_non_contiguous(m, n, dtype, device):
    # Build (n, m) then transpose to (m, n): non-contiguous with swapped strides.
    lhs = torch.rand((n, m), dtype=dtype, device=device).T
    rhs = torch.rand((n, m), dtype=dtype, device=device).T

    assert not lhs.is_contiguous()

    output = multiply_2d(lhs, rhs)
    expected = lhs * rhs

    assert torch.allclose(output, expected)
