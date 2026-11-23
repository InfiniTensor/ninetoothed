import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_silu(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)

    # TODO: Add `inplace` tests later.
    ninetoothed_output = ntops.torch.silu(input)
    reference_output = F.silu(input)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
