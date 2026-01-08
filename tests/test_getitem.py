import numpy as np
import pytest
import torch

from ninetoothed import Tensor
from tests.utils import get_available_devices


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize(
    "shape, indices",
    (
        # Basic slicing.
        ((3, 4, 2), slice(None, None, None)),
        ((3, 4, 2), slice(0, -1, 2)),
        ((2, 3, 4, 5), slice(1, None, 2)),
        # Multi-dimensional slicing.
        ((3, 4, 2), (slice(None), 1, slice(None))),
        ((3, 4, 2), (1, slice(None), None)),
        ((3, 4, 2), (slice(None, None, -1), 0, slice(1, -1))),
        ((3, 4, 2), (Ellipsis, 1)),
        ((3, 4, 2), (None, slice(None), 1, slice(None))),
        # Special cases.
        ((3, 4, 2), (slice(None), 1, slice(None))),
        ((6, 5), (slice(2, 4, None), slice(1, 4, None))),
    ),
)
def test_getitem(shape, indices, dtype, device):
    input = Tensor(shape=shape, dtype=dtype)

    output = input[indices].eval()

    # Note: We use NumPy because Torch might not support negative steps in some versions/devices.
    expected = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)[indices]

    assert np.allclose(output, expected)
