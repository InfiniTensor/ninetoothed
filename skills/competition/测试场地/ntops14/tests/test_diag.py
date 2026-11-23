import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "n, diagonal",
    [
        (1, 0),
        (5, 0),
        (5, 1),
        (5, -1),
        (5, 3),
        (5, -3),
        (10, 0),
        (10, 5),
        (10, -5),
    ],
)
def test_diag_1d(n, diagonal, dtype):
    device = "cuda"
    input = torch.randn(n, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.diag(input, diagonal)
    reference_output = torch.diag(input, diagonal)

    assert torch.allclose(ninetoothed_output, reference_output)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("diagonal", [0, 3, -3])
def test_diag_1d_empty_input(diagonal, dtype):
    device = "cuda"
    input = torch.empty((0,), dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.diag(input, diagonal)
    reference_output = torch.diag(input, diagonal)

    assert ninetoothed_output.shape == reference_output.shape
    assert torch.allclose(ninetoothed_output, reference_output)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "shape, diagonal",
    [
        ((5, 5), 0),
        ((5, 5), 1),
        ((5, 5), -1),
        ((5, 5), 4),
        ((5, 5), -4),
        ((3, 5), 0),
        ((3, 5), 1),
        ((3, 5), -1),
        ((5, 3), 0),
        ((5, 3), 1),
        ((5, 3), -1),
        ((10, 10), 0),
        ((10, 10), 3),
        ((10, 10), -3),
    ],
)
def test_diag_2d(shape, diagonal, dtype):
    device = "cuda"
    input = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.diag(input, diagonal)
    reference_output = torch.diag(input, diagonal)

    assert torch.allclose(ninetoothed_output, reference_output)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "shape, diagonal",
    [
        ((3, 5), 5),
        ((3, 5), -3),
        ((5, 3), 3),
        ((5, 3), -5),
    ],
)
def test_diag_2d_out_of_range_diagonal(shape, diagonal, dtype):
    device = "cuda"
    input = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.diag(input, diagonal)
    reference_output = torch.diag(input, diagonal)

    assert ninetoothed_output.shape == reference_output.shape
    assert torch.allclose(ninetoothed_output, reference_output)
