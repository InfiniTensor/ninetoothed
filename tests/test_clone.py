import pytest
import torch

import ninetoothed
from ninetoothed import Tensor
from tests.utils import get_available_devices


def arrangement(input, output):
    block_shape = (ninetoothed.block_size(), ninetoothed.block_size())

    input_arranged = input.tile(block_shape)
    output_arranged = output.tile(block_shape)

    return input_arranged, output_arranged


def application(input, output):
    output = input  # noqa: F841


def clone(input):
    output = torch.empty_like(input)

    tensors = (Tensor(2), Tensor(2))

    kernel = ninetoothed.make(arrangement, application, tensors)

    kernel(input, output)

    return output


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("size_n, stride_n", ((280, 300),))
@pytest.mark.parametrize("size_m, stride_m", ((641, 700),))
def test_data_ptr(size_m, size_n, stride_m, stride_n, device):
    torch.manual_seed(0)

    shape = (size_m, size_n)
    strides = (stride_m, stride_n)

    input = torch.empty_strided(shape, strides, device=device)
    input.copy_(torch.randn(shape, device=device))

    output = clone(input)

    expected = torch.clone(input)

    assert output.shape == expected.shape and torch.allclose(output, expected)
