import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()


def arrangement(x, y, z, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N):
    return (x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
            y.tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
            z.tile((BLOCK_SIZE_M, BLOCK_SIZE_N)))


def application(x, y, z):
    z = x + y  # noqa: F841


def create_2d_add_kernel():
    return ninetoothed.make(arrangement, application,
                          (Tensor(2), Tensor(2), Tensor(2)))
