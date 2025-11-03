import torch

from ninetoothed import Tensor, block_size, make
from tests.skippers import skip_if_cuda_not_available

BLOCK_SIZE = block_size()


def arrangement(x, BLOCK_SIZE=BLOCK_SIZE):
    return (x.expand((BLOCK_SIZE,)).tile((BLOCK_SIZE,)),)


def application(x):
    x


@skip_if_cuda_not_available
def test_expand():
    x = torch.empty((0,), device="cuda")

    kernel = make(arrangement, application, (Tensor(1),))

    kernel(x)
