import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(x, w, eps, y, BLOCK_SIZE=BLOCK_SIZE):
    def arrange(tensor):
        return tensor.tile((1, BLOCK_SIZE))

    return arrange(x), arrange(w), eps, arrange(y)


def application(x, w, eps, y):
    x_fp32 = ntl.cast(x, ntl.float32)
    y = x_fp32 * ntl.rsqrt(ntl.sum(x_fp32 * x_fp32) / x.shape[-1] + eps) * w


tensors = (Tensor(2), Tensor(2), Tensor(0), Tensor(2))

kernel = ninetoothed.make(arrangement, application, tensors)
