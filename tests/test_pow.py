import pytest
import torch

import ninetoothed
from ninetoothed import Tensor, block_size
from ninetoothed.language import libdevice
from tests.utils import get_available_devices


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


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("size", (44925,))
def test(size, dtype, device):
    input = torch.rand(size, dtype=dtype, device=device)
    exponent = torch.rand(size, dtype=dtype, device=device)

    output = pow(input, exponent)
    expected = torch.pow(input, exponent)

    assert torch.allclose(output, expected, equal_nan=True)
