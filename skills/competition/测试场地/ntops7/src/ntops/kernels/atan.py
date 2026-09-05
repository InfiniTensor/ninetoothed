import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def _atan_taylor_poly(x, dtype):
    """
    计算 atan(x) 的泰勒级数近似。
    有效范围：|x| <= 0.42
    在此范围内，15阶多项式足以提供 float32/double 级别的精度。
    """
    x2 = x * x
    
    # 泰勒展开系数: 1, -1/3, 1/5, -1/7, ...
    # 为了精度，我们保留足够多的小数位
    c3 = -0.333333333333
    c5 = 0.2
    c7 = -0.142857142857
    c9 = 0.111111111111
    c11 = -0.090909090909
    c13 = 0.076923076923
    c15 = -0.066666666667

    # Horner 规则计算: x * (1 + x^2 * (c3 + x^2 * (...)))
    p = ntl.cast(c15, dtype)
    p = p * x2 + ntl.cast(c13, dtype)
    p = p * x2 + ntl.cast(c11, dtype)
    p = p * x2 + ntl.cast(c9, dtype)
    p = p * x2 + ntl.cast(c7, dtype)
    p = p * x2 + ntl.cast(c5, dtype)
    p = p * x2 + ntl.cast(c3, dtype)
    
    # result = x + x^3 * p = x * (1 + x^2 * p)
    # 提公因式 x 以减少一次乘法并提高小数值的稳定性
    return x + x * x2 * p


def _atan(x, dtype):
    """
    高精度数值稳定的反正切计算。
    使用两级范围归约策略将输入映射到小区间，以保证多项式精度。
    """
    calc_dtype = dtype if dtype != ntl.float16 else ntl.float32
    
    # === 常量定义 (局部定义以避开作用域问题) ===
    PI_OVER_2 = 1.5707963267948966
    PI_OVER_4 = 0.7853981633974483
    TAN_PI_8  = 0.4142135623730950  # tan(pi/8)
    
    x_arg = ntl.cast(x, calc_dtype)
    
    # 0. 提取符号并取绝对值
    # atan(-x) = -atan(x)
    sign = ntl.where(x_arg < 0.0, -1.0, 1.0)
    abs_x = ntl.abs(x_arg)
    
    # 1. 第一级归约：处理 |x| > 1 的情况
    # 使用恒等式：atan(x) = pi/2 - atan(1/x)  (当 x > 0)
    # 如果 x > 1:
    #   val_1 = 1/x
    #   offset_1 = pi/2
    #   coef_1 = -1
    # 否则:
    #   val_1 = x
    #   offset_1 = 0
    #   coef_1 = 1
    # 当前结果表达式: offset_1 + coef_1 * atan(val_1)
    mask_gt_1 = abs_x > 1.0
    
    # 安全除法：防止 abs_x 为 0 时的除零错误（虽然此时分支不被选择）
    safe_abs_x = ntl.where(mask_gt_1, abs_x, ntl.cast(1.0, calc_dtype))
    
    val_1 = ntl.where(mask_gt_1, ntl.cast(1.0, calc_dtype) / safe_abs_x, abs_x)
    offset_1 = ntl.where(mask_gt_1, PI_OVER_2, 0.0)
    coef_1 = ntl.where(mask_gt_1, -1.0, 1.0)
    
    # 此时 val_1 在 [0, 1] 范围内
    
    # 2. 第二级归约：处理 x 接近 1 的情况
    # 使用恒等式：atan(x) = pi/4 + atan((x-1)/(x+1))
    # 阈值取 tan(pi/8) ≈ 0.414，这样归约后的值域在 [-0.29, 0.414] 之间
    # 这对泰勒级数收敛非常有利
    mask_gt_tan_pi_8 = val_1 > TAN_PI_8
    
    # 计算 (x-1)/(x+1)
    # 注意：val_1 均为非负数，分母 val_1 + 1 永远 >= 1，无除零风险
    reduced_val = (val_1 - 1.0) / (val_1 + 1.0)
    
    val_2 = ntl.where(mask_gt_tan_pi_8, reduced_val, val_1)
    offset_2 = ntl.where(mask_gt_tan_pi_8, PI_OVER_4, 0.0)
    
    # 当前结果表达式: atan(val_1) = offset_2 + atan(val_2)
    # 此时 |val_2| <= 0.4142...
    
    # 3. 多项式计算
    poly_res = _atan_taylor_poly(val_2, calc_dtype)
    
    # 4. 结果组合
    # result = sign * (offset_1 + coef_1 * (offset_2 + poly_res))
    
    # 先计算内层: atan_val_1
    atan_val_1 = ntl.cast(offset_2, calc_dtype) + poly_res
    
    # 再计算外层: abs_result
    abs_result = ntl.cast(offset_1, calc_dtype) + ntl.cast(coef_1, calc_dtype) * atan_val_1
    
    # 最后恢复符号
    final_result = ntl.cast(sign, calc_dtype) * abs_result
    
    return ntl.cast(final_result, dtype)


def application(input, output):
    """
    计算反正切函数 atan(x)
    
    参数:
    input: 输入张量，形状为 (C // block_size, block_size)
    output: 输出张量，形状为 (C // block_size, block_size)
    """
    dtype = output.dtype.dtype
    
    for i in range(input.shape[0]):
        # 获取当前块的数据
        input_block = ntl.cast(input[i], dtype)
        
        # 计算 atan(x)
        result = _atan(input_block, dtype)
        
        # 将结果存入输出
        output[i] = result


def premake(ndim, dim, dtype=None, block_size=None):
    """
    准备 atan 内核
    """
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),           # 输入张量
        Tensor(ndim, dtype=dtype),           # 输出张量
    )

    return arrangement_, application, tensors