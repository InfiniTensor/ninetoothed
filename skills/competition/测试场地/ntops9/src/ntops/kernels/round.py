import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


def application(input, output):
    output = libdevice.nearbyint(ntl.cast(input, ntl.float32))  # noqa: F841


def application_with_decimals(input, factor, inv_factor, output):
    scaled = input * ntl.cast(
        factor, input.dtype
    )  # 在 input 的原始精度下乘，匹配 torch 行为
    output = libdevice.nearbyint(ntl.cast(scaled, ntl.float32)) * inv_factor  # noqa: F841


def premake(ndim, decimals=0, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    if decimals == 0:
        tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
        return arrangement_, application, tensors
    else:
        tensors = (
            Tensor(ndim, dtype=dtype),
            Tensor(0, dtype=ninetoothed.float64),
            Tensor(0, dtype=ninetoothed.float64),
            Tensor(ndim, dtype=dtype),
        )
        return arrangement_, application_with_decimals, tensors
