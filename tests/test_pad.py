import pytest
import torch
import torch.nn.functional as F

from ninetoothed import Tensor, make
from tests.utils import get_available_devices


def arrangement(input, output, input_slices, output_slices):
    return input[input_slices], output[output_slices]


def application(input, output):
    output = input  # noqa: F841


def pad(input, pad, mode="constant", value=None):
    # This function pads a tensor.
    # Please refer to
    # [torch.nn.functional.pad](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html)
    # for more details.
    output_shape, input_slices, output_slices = _analyze_pad_config(input, pad, mode)

    value = 0 if value is None else value
    output = torch.full(output_shape, value, dtype=input.dtype, device=input.device)

    ndim = input.ndim
    kernel_config = (ndim, input_slices, output_slices)
    kernel_key = str(kernel_config)

    if kernel_key not in _kernel_cache:
        tensors = (Tensor(ndim), Tensor(ndim), input_slices, output_slices)

        _kernel_cache[kernel_key] = make(arrangement, application, tensors)

    _kernel_cache[kernel_key](input, output)

    return output


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype, atol", ((torch.float32, 1e-8), (torch.float16, 1e-3)))
@pytest.mark.parametrize("mode", ("constant",))
@pytest.mark.parametrize(
    "value", (0, 1, -1, float("-inf"), torch.pi, torch.sqrt(torch.tensor(2026)))
)
@pytest.mark.parametrize(
    "shape, pad_",
    (
        ((2026, 120712), (-100, 20, 9999, -100)),
        ((2, 3), (1, 1, 1, 2)),
        ((2, 3, 4), (1, 3, 1, 0, 0, 0)),
        ((2, 3), (-1, -1, 0, 0)),
    ),
)
def test_pad(shape, pad_, mode, value, dtype, device, atol):
    input = torch.randn(shape, dtype=dtype, device=device)

    output = pad(input, pad_, mode, value)

    expected = F.pad(input, pad_, mode, value)

    assert torch.allclose(output, expected, atol=atol)


_kernel_cache = {}


def _analyze_pad_config(input, pad, mode):
    assert mode == "constant", 'Only `"constant"` padding mode is supported.'

    ndim = input.ndim
    input_shape = list(input.shape)
    output_shape = list(input.shape)
    input_slices = []
    output_slices = []

    for i in range(ndim):
        left = pad[2 * (ndim - 1 - i)]
        right = pad[2 * (ndim - 1 - i) + 1]
        output_shape[i] += left + right

        input_start = max(0, -left)
        input_end = min(input_shape[i], input_shape[i] + right)

        output_start = max(0, left)
        output_end = output_shape[i] - max(0, right)

        input_slices.append(slice(input_start, input_end))
        output_slices.append(slice(output_start, output_end))

    input_slices = tuple(input_slices)
    output_slices = tuple(output_slices)

    return output_shape, input_slices, output_slices
