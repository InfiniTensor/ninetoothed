import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


def application(input, other, output):
    if input.dtype is ntl.float16:
        # libdevice.copysign 仅支持 float32/float64，
        # 对 float16 使用位操作保证 IEEE 754 ±0.0 语义正确
        in_bits = ntl.cast(input, ntl.uint16, bitcast=True)
        ot_bits = ntl.cast(other, ntl.uint16, bitcast=True)
        abs_bits = in_bits & 0x7FFF
        sign_bits = ot_bits & 0x8000
        result_bits = abs_bits | sign_bits
        output = ntl.cast(result_bits, ntl.float16, bitcast=True)  # noqa: F841
    else:
        # float32 / float64 直接使用 libdevice.copysign
        in_f32 = ntl.cast(input, ntl.float32)
        ot_f32 = ntl.cast(other, ntl.float32)
        result_f32 = libdevice.copysign(in_f32, ot_f32)
        output = ntl.cast(result_f32, input.dtype)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
