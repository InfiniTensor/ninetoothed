import functools
import math

import pytest
import torch

import ninetoothed
import tests.test_matmul as matmul
from ninetoothed import Tensor
from tests.utils import get_available_devices


@pytest.mark.parametrize("_device", get_available_devices())
@pytest.mark.parametrize("num_stages", (None, 3, (1, 3)))
@pytest.mark.parametrize("num_warps", (None, 4, (4, 8)))
@pytest.mark.parametrize("block_size_k", (ninetoothed.block_size(), 64))
@pytest.mark.parametrize("block_size_n", (ninetoothed.block_size(), 64))
@pytest.mark.parametrize("block_size_m", (ninetoothed.block_size(), 64))
def test_auto_tuning_generation(
    block_size_m, block_size_n, block_size_k, num_warps, num_stages, _device
):
    auto_tuning_should_be_disabled = (
        isinstance(block_size_m, int)
        and isinstance(block_size_n, int)
        and isinstance(block_size_k, int)
        and not isinstance(num_warps, tuple)
        and not isinstance(num_stages, tuple)
    )

    arrangement = functools.partial(
        matmul.arrangement,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
    )

    application = matmul.application

    tensors = (Tensor(2), Tensor(2), Tensor(2))

    kernel = ninetoothed.make(
        arrangement, application, tensors, num_warps=num_warps, num_stages=num_stages
    )

    source_file = kernel._source

    with open(source_file) as f:
        contents = f.read()

        if auto_tuning_should_be_disabled:
            assert "application_with_auto_tuning" not in contents
            assert "num_warps=" in contents and "num_stages=" in contents
        else:
            assert "application_with_auto_tuning" in contents


@pytest.mark.parametrize("_device", get_available_devices())
def test_arrangement_returning_a_single_tensor(_device):
    def arrangement(x):
        return x.tile((ninetoothed.block_size(),))

    def application(x):
        x

    ninetoothed.make(arrangement, application, (Tensor(1),))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("num_indices", (10,))
@pytest.mark.parametrize("num_cols", (128,))
@pytest.mark.parametrize("num_rows", (1024,))
def test_squeezing_the_innermost_level(num_rows, num_cols, num_indices, device):
    torch.manual_seed(0)

    def arrangement(input, indices, output):
        input_arranged = input.tile((1, -1)).squeeze(1)
        input_arranged.dtype = input_arranged.dtype.squeeze(0)

        indices_arranged = indices.tile((1,))

        output_arranged = (
            output.tile((1, -1))
            .tile((-1, -1))
            .squeeze(0)
            .expand((input_arranged.shape[0],))
        )
        output_arranged.dtype = output_arranged.dtype.squeeze(1)
        output_arranged.dtype.dtype = output_arranged.dtype.dtype.squeeze(0)

        return input_arranged, indices_arranged, output_arranged

    def application(input, index, output):
        row = index

        output[row] = input

    shape_options = {"constexpr": True}

    tensors = (
        Tensor(2, shape_options=shape_options),
        Tensor(1, shape_options=shape_options),
        Tensor(2, shape_options=shape_options),
    )

    kernel = ninetoothed.make(arrangement, application, tensors)

    input = torch.randn((num_indices, num_cols), device=device)
    indices = torch.randint(0, num_rows, (num_indices,), device=device)
    output = torch.empty((num_rows, num_cols), device=device)
    expected = output.clone()

    kernel(input, indices, output)

    for i in range(num_indices):
        expected[indices[i], :] = input[i, :]

    assert output.shape == expected.shape and torch.allclose(output, expected)


@pytest.mark.parametrize("device", get_available_devices())
def test_unsqueezing_the_outermost_level(device):
    def arrangement(x):
        return x.tile((ninetoothed.block_size(),)).unsqueeze(0)

    def application(x):
        x

    kernel = ninetoothed.make(arrangement, application, (Tensor(1),))

    kernel(torch.randn((0,), device=device))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("size", (1086,))
def test_non_int_constexpr(size, device):
    def arrangement(input, other, output, block_size=ninetoothed.block_size()):
        return input.tile((block_size,)), other, output.tile((block_size,))

    def application(input, other, output):
        output = input + other  # noqa: F841

    tensors = (Tensor(1), Tensor(0, constexpr=True), Tensor(1))

    kernel = ninetoothed.make(arrangement, application, tensors)

    input = torch.randn((size,), device=device)
    other = math.pi
    output = torch.empty_like(input)
    expected = input + other

    kernel(input, other, output)

    assert output.shape == expected.shape and torch.allclose(output, expected)
