import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()


def arrangement(x, y, z, BLOCK_SIZE=BLOCK_SIZE):
    return (x.tile((BLOCK_SIZE,)),
            y.tile((BLOCK_SIZE,)),
            z.tile((BLOCK_SIZE,)))


def application(x, y, z):
    z = x + y  # noqa: F841


def create_add_kernel():
    return ninetoothed.make(arrangement, application,
                          (Tensor(1), Tensor(1), Tensor(1)))
