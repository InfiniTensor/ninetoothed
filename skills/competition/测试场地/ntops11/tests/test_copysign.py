import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_copysign(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    with torch.no_grad():
        ninetoothed_output = ntops.torch.copysign(input, other)
        reference_output = torch.copysign(input, other)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_copysign_signed_zero(dtype):
    device = "cuda"
    input = torch.tensor([0.0, -0.0, 0.0, -0.0, 1.5, -2.5], dtype=dtype, device=device)
    other = torch.tensor([1.0, 1.0, -1.0, -1.0, -3.0, 4.0], dtype=dtype, device=device)

    with torch.no_grad():
        ninetoothed_output = ntops.torch.copysign(input, other)
        reference_output = torch.copysign(input, other)

    # IEEE 754: -0.0 == +0.0，所以用 signbit 区分
    assert torch.signbit(ninetoothed_output).tolist() == torch.signbit(
        reference_output
    ).tolist()
    assert torch.allclose(
        ninetoothed_output.abs(), reference_output.abs(), rtol=1e-3, atol=1e-3
    )


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_copysign_special_values(dtype):
    device = "cuda"
    input = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan")],
        dtype=dtype,
        device=device,
    )
    other = torch.tensor(
        [-1.0, 1.0, -2.0, 2.0, -1.0, 1.0, -1.0], dtype=dtype, device=device
    )

    with torch.no_grad():
        ninetoothed_output = ntops.torch.copysign(input, other)
        reference_output = torch.copysign(input, other)

    assert torch.signbit(ninetoothed_output).tolist() == torch.signbit(
        reference_output
    ).tolist()

    # NaN != NaN，所以只对比非 NaN 位置
    nan_mask = torch.isnan(reference_output)
    assert torch.isnan(ninetoothed_output)[nan_mask].all()
    assert torch.allclose(
        ninetoothed_output[~nan_mask],
        reference_output[~nan_mask],
        rtol=1e-3,
        atol=1e-3,
    )
