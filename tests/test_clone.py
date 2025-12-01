import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from tests.utils import get_available_devices


def arrangement(input, output):
    block_shape = (ninetoothed.block_size(), ninetoothed.block_size())

    input_arranged = input.tile(block_shape)
    output_arranged = output.tile(block_shape)

    return input_arranged, output_arranged


def application(input, output):
    output = input  # noqa: F841


def application_1(input, output):
    output = ntl.load(  # noqa: F841
        input.data_ptr()
        + input.offsets(0)[:, None] * input.stride(0)
        + input.offsets(1)[None, :] * input.stride(1)
    )


applications = (application, application_1)


def clone(input, *, impl_id=0):
    output = torch.empty_like(input)

    tensors = (Tensor(2), Tensor(2))

    kernel = ninetoothed.make(arrangement, applications[impl_id], tensors)

    kernel(input, output)

    return output


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("size_n, stride_n", ((280, 300),))
@pytest.mark.parametrize("size_m, stride_m", ((641, 700),))
@pytest.mark.parametrize("impl_id", range(len(applications)))
def test_data_ptr(impl_id, size_m, size_n, stride_m, stride_n, device):
    torch.manual_seed(0)

    shape = (size_m, size_n)
    strides = (stride_m, stride_n)

    input = torch.empty_strided(shape, strides, device=device)
    input.copy_(torch.randn(shape, device=device))

    output = clone(input, impl_id=impl_id)

    expected = torch.clone(input)

    assert output.shape == expected.shape and torch.allclose(output, expected)
