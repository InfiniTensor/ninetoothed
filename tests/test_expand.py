import pytest
import torch

from ninetoothed import Tensor, block_size, make
from tests.utils import get_available_devices

BLOCK_SIZE = block_size()


def arrangement(x, BLOCK_SIZE=BLOCK_SIZE):
    return (x.expand((BLOCK_SIZE,)).tile((BLOCK_SIZE,)),)


def application(x):
    x


@pytest.mark.parametrize("device", get_available_devices())
def test_expand(device):
    x = torch.empty((0,), device=device)

    kernel = make(arrangement, application, (Tensor(1),))

    kernel(x)
