import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed.language import libdevice
from ninetoothed import Tensor
from ninetoothed import Symbol


def arrangement(input, output, norm_type, kernel_size_val, kernel_size, stride, block_size, ceil_mode):
    if block_size is None:
        block_size = ninetoothed.block_size()

    # input: (N, C, L_in) output: (N, C, L_out)

    input_arranged = input.tile((1, 1, kernel_size), (1, 1, stride), floor_mode=not ceil_mode)
    # => (N, C, L_out), dtype=(1, 1, k)
    input_arranged = input_arranged.ravel()
    # => (N, C, L_out, 1, 1, k)
    input_arranged = input_arranged.flatten(end_dim=3).flatten(start_dim=1)
    # => (N*C*L_out, k)
    # k 的找到最近的 2 的倍数
    nearest_pow2 = 1 << (kernel_size - 1).bit_length()
    input_arranged = input_arranged.tile((1, nearest_pow2))
    # => (..., k // nearest_pow2 = 1), dtype=(1, nearest_pow2)
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    # => (..., 1), dtype=(nearest_pow2, )
    input_arranged = input_arranged.tile((block_size, -1))
    # => (..., 1), dtype=(block_size, 1), dtype=(nearest_pow2, )
    input_arranged.dtype = input_arranged.dtype.ravel().squeeze(1)
    # => (..., 1), dtype=(block_size, nearest_pow2)

    output_arranged = output.tile((1, 1, 1)) 
    # => (N, C, L_out), dtype=(1, 1, 1)
    output_arranged = output_arranged.ravel()
    # => (N, C, L_out, 1, 1, 1)
    output_arranged = output_arranged.flatten(end_dim=3).flatten(start_dim=1)
    # => (N*C*L_out, 1)
    output_arranged = output_arranged.tile((block_size, -1)) 
    # => (..., 1), dtype=(block_size, 1)
    output_arranged.dtype = output_arranged.dtype.squeeze(1) 
    # => (..., 1), dtype=(block_size, )

    return input_arranged, output_arranged, norm_type, kernel_size_val


def _pow(x, norm, dtype):
    pow_dtype = dtype if dtype != ntl.float16 else ntl.float32
    return ntl.cast(libdevice.pow(ntl.cast(x, pow_dtype), norm), dtype)

def application(input, output, norm_type, kernel_size):
    # input: (block_size, nearest_pow2)
    # output: (block_size)
    dtype = input.dtype
    mask = input < 1e20
    cnt = ntl.sum(ntl.cast(mask, ntl.int32), axis=1)
    input_masked = ntl.where(~mask, 0, input)
    x_pow = _pow(input_masked, norm_type, dtype)
    acc_sim = ntl.sum(x_pow, 1) / cnt * kernel_size
    output = _pow(acc_sim, 1.0 / norm_type, dtype)


def premake(ndim, kernel_size, stride, ceil_mode=False, dtype=None, block_size=None):
    arrangement_ = functools.partial(
        arrangement,
        kernel_size=kernel_size,
        stride=stride,
        block_size=block_size,
        ceil_mode=ceil_mode,
    )

    tensors = (
        Tensor(ndim, dtype=dtype, other=float("inf")),  # input
        Tensor(ndim, dtype=dtype),  # output
        Tensor(0, dtype=dtype),      # norm_type
        Tensor(0, dtype=dtype, constexpr=True),      # kernel_size 
    )

    return arrangement_, application, tensors
