import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("ninetoothed")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from softmax import softmax  # noqa: E402


def get_available_devices():
    devices = []

    if torch.cuda.is_available():
        devices.append("cuda")

    return tuple(devices)


pytestmark = pytest.mark.skipif(
    not get_available_devices(),
    reason="CUDA is required to run NineToothed softmax kernels",
)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("m, n", ((1823, 781), (37, 129), (4, 1024)))
def test_softmax_matches_pytorch(m, n, dtype, device):
    input = torch.randn((m, n), dtype=dtype, device=device)

    output = softmax(input)
    expected = torch.softmax(input, dim=-1)

    assert output.shape == expected.shape
    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
def test_softmax_is_numerically_stable(dtype, device):
    input = torch.randn((32, 257), dtype=dtype, device=device) * 20

    output = softmax(input)
    expected = torch.softmax(input, dim=-1)

    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(
        output.sum(dim=-1), torch.ones(32, dtype=dtype, device=device), atol=1e-6
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_softmax_rejects_non_2d(device):
    input = torch.randn((2, 3, 4), device=device)

    with pytest.raises(NotImplementedError):
        softmax(input)
