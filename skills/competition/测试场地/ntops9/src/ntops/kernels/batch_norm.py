import functools
import math

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def application(input, weight, bias, eps, output, num_normalized_elements):
    # 使用 E[x^2] - E[x]^2 公式计算方差，避免显式处理 Padding Mask
    # 因为 Padding 处 input 为 0，0 的平方也是 0，不会污染 sum 和 sum_sq
    
    _sum = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    _sum_sq = ntl.zeros(input.dtype.shape, dtype=ntl.float32)

    # Pass 1: 计算 Sum 和 Sum of Squares
    for i in range(input.shape[0]):
        val = ntl.cast(input[i], ntl.float32)
        _sum += val
        _sum_sq += val * val

    mean = ntl.sum(_sum, 0) / num_normalized_elements
    mean_sq = ntl.sum(_sum_sq, 0) / num_normalized_elements
    
    # Var = E[x^2] - (E[x])^2
    var = mean_sq - mean * mean
    # 确保方差非负 (处理数值误差)
    var = ntl.maximum(var, 0.0)

    std = ntl.sqrt(var + eps)

    # Pass 2: 归一化并输出
    # 这里的 store 操作通常会被编译器根据 Tensor 形状自动 Mask 掉越界部分
    for i in range(input.shape[0]):
        output[i] = (ntl.cast(input[i], ntl.float32) - mean) / std * weight[i] + bias[i]


def premake(ndim, reduction_dims, num_elements, dtype=None, block_size=None):
    # reduction_dims 指定了需要在哪些维度上进行规约
    arrangement_ = functools.partial(arrangement, dim=reduction_dims, block_size=block_size)

    tensors = (
        Tensor(ndim, other=0, dtype=dtype),           # Input (other=0 确保 padding 读入 0)
        Tensor(ndim, dtype=dtype),                    # Weight
        Tensor(ndim, dtype=dtype),                    # Bias
        Tensor(0, dtype=dtype),                       # eps
        Tensor(ndim, dtype=dtype),                    # Output
        Tensor(0, dtype=dtype, constexpr=True, value=num_elements),
    )

    return arrangement_, application, tensors