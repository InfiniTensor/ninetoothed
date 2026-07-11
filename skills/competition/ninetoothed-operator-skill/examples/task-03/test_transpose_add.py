import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("ninetoothed")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from transpose_add import transpose_add  # noqa: E402


def get_available_devices():
    devices = []

    if torch.cuda.is_available():
        devices.append("cuda")

    return tuple(devices)


pytestmark = pytest.mark.skipif(
    not get_available_devices(),
    reason="CUDA is required to run NineToothed transpose_add kernels",
)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("m, n", ((257, 129), (64, 512)))
def test_transpose_add_contiguous(m, n, dtype, device):
    input = torch.randn((m, n), dtype=dtype, device=device)
    bias = torch.randn((n, m), dtype=dtype, device=device)

    output = transpose_add(input, bias)
    expected = input.transpose(0, 1) + bias

    assert output.shape == expected.shape
    assert torch.allclose(output, expected)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
def test_transpose_add_non_contiguous_input(dtype, device):
    base = torch.randn((96, 80), dtype=dtype, device=device)
    input = base[3:67:2, 5:73:3]
    bias = torch.randn((input.shape[1], input.shape[0]), dtype=dtype, device=device)

    assert not input.is_contiguous()
    assert input.storage_offset() > 0

    output = transpose_add(input, bias)
    expected = input.transpose(0, 1) + bias

    assert output.shape == expected.shape
    assert torch.allclose(output, expected)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
def test_transpose_add_empty_strided_input(dtype, device):
    shape = (37, 29)
    input = torch.empty_strided(shape, (41, 2), dtype=dtype, device=device)
    input.copy_(torch.randn(shape, dtype=dtype, device=device))
    bias = torch.randn((shape[1], shape[0]), dtype=dtype, device=device)

    output = transpose_add(input, bias)
    expected = input.transpose(0, 1) + bias

    assert output.shape == expected.shape
    assert torch.allclose(output, expected)


@pytest.mark.parametrize("device", get_available_devices())
def test_transpose_add_rejects_bad_bias_shape(device):
    input = torch.randn((4, 5), device=device)
    bias = torch.randn((4, 5), device=device)

    with pytest.raises(ValueError):
        transpose_add(input, bias)
