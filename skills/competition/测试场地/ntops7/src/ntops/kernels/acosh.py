import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def _sqrt(x, dtype):
    """数值稳定的平方根计算，特别处理float16精度"""
    sqrt_dtype = dtype if dtype != ntl.float16 else ntl.float32
    return ntl.cast(ntl.sqrt(ntl.cast(x, sqrt_dtype)), dtype)


def _log(x, dtype):
    """数值稳定的对数计算，特别处理float16精度"""
    log_dtype = dtype if dtype != ntl.float16 else ntl.float32
    return ntl.cast(ntl.log(ntl.cast(x, log_dtype)), dtype)


def application(input, output):
    """
    计算反双曲余弦函数 acosh(x) = ln(x + sqrt(x² - 1))
    
    参数:
    input: 输入张量，形状为 (C // block_size, block_size)
    output: 输出张量，形状为 (C // block_size, block_size)
    
    数值稳定性考虑:
    1. 当x接近1时，x² - 1接近0，使用(x-1)(x+1)形式避免精度损失
    2. 当x很大时，避免x²溢出，使用代数变换
    3. float16特殊处理，提升到float32计算
    """
    dtype = output.dtype.dtype
    
    for i in range(input.shape[0]):
        # 获取当前块的数据
        input_block = ntl.cast(input[i], dtype)
        
        # 数值稳定的acosh计算
        # acosh(x) = ln(x + sqrt(x² - 1))
        
        # 处理x接近1的情况：x² - 1 = (x-1)(x+1)
        # 这样可以避免当x接近1时的精度损失
        x_minus_one = input_block - ntl.cast(1.0, dtype)
        x_plus_one = input_block + ntl.cast(1.0, dtype)
        
        # 计算 sqrt(x² - 1) = sqrt((x-1)(x+1))
        sqrt_term = _sqrt(x_minus_one * x_plus_one, dtype)
        
        # 计算 x + sqrt(x² - 1)
        # 当x很大时，这可能会导致数值问题，但acosh的定义域x≥1
        sum_term = input_block + sqrt_term
        
        # 最终计算 ln(x + sqrt(x² - 1))
        result = _log(sum_term, dtype)
        
        # 处理边界情况：当x < 1时，acosh未定义，返回NaN
        # 当x == 1时，acosh(1) = 0
        result = ntl.where(
            input_block < ntl.cast(1.0, dtype),
            ntl.cast(float("nan"), dtype),
            ntl.where(
                input_block == ntl.cast(1.0, dtype),
                ntl.cast(0.0, dtype),
                result
            )
        )
        
        # 将结果存入输出
        output[i] = result


def premake(ndim, dim, dtype=None, block_size=None):
    """
    准备acosh内核
    
    参数:
    ndim: 输入张量的维度
    dim: 要计算acosh的维度
    dtype: 数据类型
    block_size: 分块大小，用于优化内存访问
    
    返回:
    arrangement_: 张量排列函数
    application: 计算函数
    tensors: 输入输出张量描述
    """
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),           # 输入张量
        Tensor(ndim, dtype=dtype),           # 输出张量
    )

    return arrangement_, application, tensors