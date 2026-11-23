import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize("dim_is_none", (False, True))
@pytest.mark.parametrize("keepdim", (False, True))
@pytest.mark.parametrize(
    "interpolation", ("linear", "lower", "higher", "nearest", "midpoint")
)
@pytest.mark.parametrize(*generate_arguments())
def test_quantile(
    shape, dtype, device, rtol, atol, dim_is_none, keepdim, interpolation
):
    if dtype == torch.float16:
        pytest.skip(reason="`torch.quantile` does not support float16")

    input = torch.randn(shape, dtype=dtype, device=device)
    qtype = random.choice(("tensor", "scalar"))
    if qtype == "tensor":
        q_size = random.randint(1, 5)
        q = torch.rand(q_size, dtype=dtype, device=device)
    else:
        q = random.random()
    if dim_is_none:
        dim = None
    else:
        dim = random.randint(0, input.ndim - 1)

    ninetoothed_output = ntops.torch.quantile(
        input, q, dim=dim, keepdim=keepdim, interpolation=interpolation
    )
    reference_output = torch.quantile(
        input, q, dim=dim, keepdim=keepdim, interpolation=interpolation
    )

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
