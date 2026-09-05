import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _check(shape, dtype, value=1.0, atol=None, rtol=None):
    if atol is None:
        atol = 9.77e-4 if dtype == torch.float16 else 1.22e-4
    if rtol is None:
        rtol = atol

    device = "cuda"
    input = torch.randn(shape, dtype=dtype, device=device)
    tensor1 = torch.randn(shape, dtype=dtype, device=device)
    tensor2 = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.addcmul(
        input, tensor1, tensor2, value=value
    )
    reference_output = torch.addcmul(input, tensor1, tensor2, value=value)

    diff = (ninetoothed_output - reference_output).abs()
    max_diff = diff.max().item()
    assert torch.allclose(
        ninetoothed_output, reference_output, rtol=rtol, atol=atol
    ), f"max_diff={max_diff}"


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((128, 128), torch.float32),
        ((64, 64), torch.float32),
    ],
)
def test_addcmul_basic(shape, dtype):
    _check(shape, dtype)


@skip_if_cuda_not_available
def test_addcmul_large():
    _check((4096, 4096), torch.float16, value=3.0, atol=9.77e-4, rtol=9.77e-4)


@skip_if_cuda_not_available
def test_addcmul_fp16():
    _check((1024, 1024), torch.float16, value=10.0, atol=9.77e-4, rtol=9.77e-4)
