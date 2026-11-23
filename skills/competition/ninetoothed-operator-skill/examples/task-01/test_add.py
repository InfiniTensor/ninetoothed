import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("ninetoothed")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from add import add  # noqa: E402


def get_available_devices():
    devices = []

    if torch.cuda.is_available():
        devices.append("cuda")

    return tuple(devices)


pytestmark = pytest.mark.skipif(
    not get_available_devices(),
    reason="CUDA is required to run NineToothed add kernels",
)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("size", (98432,))
def test_add_1d_same_shape(size, dtype, device):
    lhs = torch.rand(size, dtype=dtype, device=device)
    rhs = torch.rand(size, dtype=dtype, device=device)

    output = add(lhs, rhs)
    expected = lhs + rhs

    assert output.shape == expected.shape
    assert torch.allclose(output, expected)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("n", (512,))
@pytest.mark.parametrize("m", (512,))
def test_add_2d_same_shape(m, n, dtype, device):
    lhs = torch.rand((m, n), dtype=dtype, device=device)
    rhs = torch.rand((m, n), dtype=dtype, device=device)

    output = add(lhs, rhs)
    expected = lhs + rhs

    assert output.shape == expected.shape
    assert torch.allclose(output, expected)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("n", (512,))
@pytest.mark.parametrize("m", (512,))
def test_add_2d_1d_broadcast(m, n, dtype, device):
    lhs = torch.rand((m, n), dtype=dtype, device=device)
    rhs = torch.rand(n, dtype=dtype, device=device)

    output = add(lhs, rhs)
    expected = lhs + rhs

    assert output.shape == expected.shape
    assert torch.allclose(output, expected)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("n", (512,))
@pytest.mark.parametrize("m", (512,))
def test_add_2d_row_broadcast(m, n, dtype, device):
    lhs = torch.rand((m, n), dtype=dtype, device=device)
    rhs = torch.rand((1, n), dtype=dtype, device=device)

    output = add(lhs, rhs)
    expected = lhs + rhs

    assert output.shape == expected.shape
    assert torch.allclose(output, expected)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("n", (512,))
@pytest.mark.parametrize("m", (512,))
def test_add_col_broadcast(m, n, dtype, device):
    lhs = torch.rand((1, n), dtype=dtype, device=device)
    rhs = torch.rand((m, n), dtype=dtype, device=device)

    output = add(lhs, rhs)
    expected = lhs + rhs

    assert output.shape == expected.shape
    assert torch.allclose(output, expected)
