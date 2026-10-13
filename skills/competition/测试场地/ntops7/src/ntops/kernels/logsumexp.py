import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def _exp(x, dtype):
    exp_dtype = dtype if dtype != ntl.float16 else ntl.float32
    return ntl.cast(ntl.exp(ntl.cast(x, exp_dtype)), dtype)

def _log(x, dtype):
    log_dtype = dtype if dtype != ntl.float16 else ntl.float32
    return ntl.cast(ntl.log(ntl.cast(x, log_dtype)), dtype)

def application(input, output):
    # input&output: (C // block_size, )
    # input.dtype: (block_size, )
    dtype = output.dtype.dtype
    prev_max = ntl.cast(float("-inf"), dtype)
    denominator = ntl.cast(0, dtype)

    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], dtype)
        curr_max = ntl.cast(ntl.maximum(prev_max, ntl.max(input_i)), dtype)
        input_max_diff_exp = _exp(input_i - curr_max, dtype)
        prev_curr_max_diff_exp = _exp(prev_max - curr_max, dtype)
        denominator = denominator * prev_curr_max_diff_exp + ntl.sum(input_max_diff_exp)
        prev_max = curr_max

    output[0] = prev_max + _log(denominator, dtype)


def premake(ndim, dim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    tensors = (
        Tensor(
            ndim, dtype=dtype, other=float("-inf"), shape_options={"constexpr": True}
        ),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
