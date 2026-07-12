import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed.language import libdevice
from ninetoothed import Tensor
from ninetoothed import Symbol


def _pow(x, norm, dtype):
    pow_dtype = dtype if dtype != ntl.float16 else ntl.float32
    return ntl.cast(libdevice.pow(ntl.cast(x, pow_dtype), norm), dtype)

def arrangement_ceil_mode(
    *tensors,
    kernel_size_d,
    kernel_size_h,
    kernel_size_w,
    stride_d,
    stride_h,
    stride_w,
    block_size,
    ceil_mode,
):
    """ceil_mode 下的 arrangement, 需要额外传入 kernel_size_flatted"""
    input, output, norm_type, kernel_size_flatted = tensors
    input_arranged, output_arranged, norm_type = arrangement(
        input,
        output,
        norm_type,
        kernel_size_d,
        kernel_size_h,
        kernel_size_w,
        stride_d,
        stride_h,
        stride_w,
        block_size,
        ceil_mode,
    )
    return input_arranged, output_arranged, norm_type, kernel_size_flatted



def application_ceil_mode(input, output, norm_type, kernel_size_flatted):
    # input: (block_size, nearest_pow2)
    # output: (block_size, )
    # INFO: 下面的内容同时适用于 lp_pool2d 和 lp_pool3d
    # 这里 torch 实现与文档上的不一致，文档上描述的是 sum(windows^p)^(1/p)
    # 实际上 torch 的实现是 mean(windows^p) * (kernel_size_h * kernel_size_w))^(1/p)
    # 这在 strides=kernel_size 时的结果是一致的，但是在 strides!=kernel_size && ceil_mode=True 时会有差异
    # 主要体现在边界处理上, torch 的算法会放大边界处的值，因为边界处的窗口内有效元素个数少于 kernel_size_h * kernel_size_w
    # 下面给出了两种不同的实现
    # 这是补 0 的实现 (要使用这种实现，请将input的默认值修改为 0)
    # dtype = input.dtype
    # x_pow = _pow(input, norm_type, dtype)
    # acc = ntl.sum(x_pow, axis=0)
    # output = _pow(acc, 1.0 / norm_type, dtype)
    
    # 我把 ceil_mode 和普通的实现区分开来了
    # 为了通过测试，下面使用的是与 torch 实现一致的版本
    dtype = input.dtype
    mask = input < 1e20
    cnt = ntl.sum(ntl.cast(mask, ntl.int32), axis=1)
    input_masked = ntl.where(~mask, 0, input)
    x_pow = _pow(input_masked, norm_type, dtype)
    acc_sim = ntl.sum(x_pow, 1) / cnt * kernel_size_flatted
    output = _pow(acc_sim, 1.0 / norm_type, dtype)
    

def premake_ceil_mode(ndim, kernel_size_d, kernel_size_h, kernel_size_w, stride_d, stride_h, stride_w, block_size=None, ceil_mode=False, dtype=None):
    arrangement_ = functools.partial(
        arrangement_ceil_mode,
        kernel_size_d=kernel_size_d,
        kernel_size_h=kernel_size_h,
        kernel_size_w=kernel_size_w,
        stride_d=stride_d,
        stride_h=stride_h,
        stride_w=stride_w,
        block_size=block_size,
        ceil_mode=ceil_mode,
    )

    tensors = (
        Tensor(ndim, dtype=dtype, other=float("inf")),     # input
        Tensor(ndim, dtype=dtype),              # output
        Tensor(0, dtype=dtype),                 # norm_type
        Tensor(0, dtype=dtype),                 # kernel_size_flatted
    )

    return arrangement_, application_ceil_mode, tensors



def arrangement(input, output, norm_type, kernel_size_d, kernel_size_h, kernel_size_w, stride_d, stride_h, stride_w, block_size, ceil_mode):
    if block_size is None:
        block_size = ninetoothed.block_size()

    # input: (N, C, D_in, H_in, W_in) output: (N, C, D_out, H_out, W_out)
    # ref. example 里的 max_pool2d arrangement

    input_arranged = input.tile((1, 1, kernel_size_d, kernel_size_h, kernel_size_w), (1, 1, stride_d, stride_h, stride_w), floor_mode=not ceil_mode)
    # => (N, C, D_out, H_out, W_out), dtype=(1, 1, k_d, k_h, k_w)
    input_arranged = input_arranged.ravel()
    # => (N, C, D_out, H_out, W_out, 1, 1, k_d, k_h, k_w)
    input_arranged = input_arranged.flatten(end_dim=5).flatten(start_dim=1)
    # => (N*C*D_out*H_out*W_out, k_d*k_h*k_w)

    # k_d*k_h*k_w 的找到最近的 2 的倍数
    nearest_pow2 = 1 << (kernel_size_d * kernel_size_h * kernel_size_w - 1).bit_length()
    input_arranged = input_arranged.tile((1, nearest_pow2))
    # => (..., k_d*k_h*k_w // nearest_pow2 = 1), dtype=(1, nearest_pow2)
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    # => (..., 1), dtype=(nearest_pow2, )
    input_arranged = input_arranged.tile((block_size, -1))
    # => (..., 1), dtype=(block_size, 1), dtype=(nearest_pow2, )
    input_arranged.dtype = input_arranged.dtype.ravel().squeeze(1)
    # => (..., 1), dtype=(block_size, nearest_pow2)

    output_arranged = output.tile((1, 1, 1, 1, 1)) 
    # => (N, C, D_out, H_out, W_out), dtype=(1, 1, 1, 1, 1)
    output_arranged = output_arranged.ravel()
    # => (N, C, D_out, H_out, W_out, 1, 1, 1, 1)
    output_arranged = output_arranged.flatten(end_dim=5).flatten(start_dim=1)
    # => (N*C*D_out*H_out*W_out, 1)
    output_arranged = output_arranged.tile((block_size, -1)) 
    # => (..., 1), dtype=(block_size, 1)
    output_arranged.dtype = output_arranged.dtype.squeeze(1) 
    # => (..., 1), dtype=(block_size, )

    return input_arranged, output_arranged, norm_type

def application(input, output, norm_type):
    # input: (block_size, nearest_pow2)
    # output: (block_size, )
    dtype = input.dtype
    x_pow = _pow(input, norm_type, dtype)
    acc = ntl.sum(x_pow, axis=1)
    output = _pow(acc, 1.0 / norm_type, dtype)


def premake(ndim, kernel_size_d, kernel_size_h, kernel_size_w, stride_d, stride_h, stride_w, block_size=None, ceil_mode=False, dtype=None):
    arrangement_ = functools.partial(
        arrangement,
        kernel_size_d=kernel_size_d,
        kernel_size_h=kernel_size_h,
        kernel_size_w=kernel_size_w,
        stride_d=stride_d,
        stride_h=stride_h,
        stride_w=stride_w,
        block_size=block_size,
        ceil_mode=ceil_mode,
    )

    tensors = (
        Tensor(ndim, dtype=dtype, other=0),     # input
        Tensor(ndim, dtype=dtype),              # output
        Tensor(0, dtype=dtype),                 # norm_type
    )

    return arrangement_, application, tensors
