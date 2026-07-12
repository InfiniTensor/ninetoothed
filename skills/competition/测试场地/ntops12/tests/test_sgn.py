import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize("is_complex", (False, True))
@pytest.mark.parametrize(*generate_arguments())
def test_sgn(shape, is_complex, dtype, device, rtol, atol):
    if dtype == torch.float16 or not is_complex:
        input_tensor = torch.randn(shape, dtype=dtype, device=device)
    else:
        real_part = torch.randn(shape, dtype=dtype.to_real(), device=device)
        imag_part = torch.randn(shape, dtype=dtype.to_real(), device=device)
        input_tensor = torch.complex(real_part, imag_part)

    ninetoothed_output = ntops.torch.sgn(input_tensor)
    reference_output = torch.sgn(input_tensor)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
