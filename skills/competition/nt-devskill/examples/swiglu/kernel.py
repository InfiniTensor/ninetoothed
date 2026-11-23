import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(a, b, c, BLOCK_SIZE=BLOCK_SIZE):
    return a.tile((BLOCK_SIZE,)), b.tile((BLOCK_SIZE,)), c.tile((BLOCK_SIZE,))


def application(a, b, c):
    # SwiGLU: c = a * (b * sigmoid(b))
    gate = b * ntl.sigmoid(ntl.cast(b, ntl.float32))
    c = a * gate


tensors = (Tensor(1), Tensor(1), Tensor(1))

kernel = ninetoothed.make(arrangement, application, tensors)
