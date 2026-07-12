import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

def _cast_to_f32(x, dtype):
    """
    为了保证累加精度，如果是 float16 则转为 float32 计算
    """
    return ntl.cast(x, ntl.float32) if dtype == ntl.float16 else x

def arrangement(input, output, kernel_size_flatted, kernel_size_h, kernel_size_w, stride_h, stride_w, block_size):
    if block_size is None:
        block_size = ninetoothed.block_size()

    # input: (N, C, H_in, W_in) 
    # output: (N, C, H_out, W_out)

    # 1. 窗口切分
    input_arranged = input.tile(
        (1, 1, kernel_size_h, kernel_size_w), 
        (1, 1, stride_h, stride_w)
    )
    # => (N, C, H_out, W_out), dtype=(1, 1, k_h, k_w)

    # 2. 展平与重排
    input_arranged = input_arranged.ravel()
    # => (N, C, H_out, W_out, 1, 1, k_h, k_w)
    
    input_arranged = input_arranged.flatten(end_dim=4).flatten(start_dim=1)
    # => (N*C*H_out*W_out, k_h*k_w)

    # 3. Padding 到最近的 2 的幂次 (用于规约)
    # 这里的 padding 值由 premake 中的 other=0 决定
    nearest_pow2 = 1 << (kernel_size_h * kernel_size_w - 1).bit_length()
    input_arranged = input_arranged.tile((1, nearest_pow2))
    # => (..., 1), dtype=(1, nearest_pow2)
    
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    input_arranged = input_arranged.tile((block_size, -1))
    input_arranged.dtype = input_arranged.dtype.ravel().squeeze(1)
    # => (..., 1), dtype=(block_size, nearest_pow2)

    # 4. Output 对齐
    output_arranged = output.tile((1, 1, 1, 1))
    output_arranged = output_arranged.ravel()
    output_arranged = output_arranged.flatten(end_dim=4).flatten(start_dim=1)
    output_arranged = output_arranged.tile((block_size, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)

    return input_arranged, output_arranged, kernel_size_flatted

def application(input, output, kernel_size_flatted):
    # input: (block_size, nearest_pow2)
    # output: (block_size, )
    # kernel_size_flatted: scalar tensor (k_h * k_w)

    dtype = input.dtype
    
    # 转为高精度进行 Sum
    val = _cast_to_f32(input, dtype)
    
    # 求和 (Axis 1 对应 nearest_pow2 维度)
    # 这里的 0 填充不会影响 Sum 结果
    acc = ntl.sum(val, axis=1)
    
    # 求平均： Sum / Area
    # 注意：kernel_size_flatted 是实际的窗口大小，不是 nearest_pow2
    res = acc / ntl.cast(kernel_size_flatted, acc.dtype)
    
    # 转回原类型
    output = ntl.cast(res, dtype)

def premake(ndim, kernel_size_h, kernel_size_w, stride_h, stride_w, block_size=None, dtype=None):
    arrangement_ = functools.partial(
        arrangement,
        kernel_size_h=kernel_size_h,
        kernel_size_w=kernel_size_w,
        stride_h=stride_h,
        stride_w=stride_w,
        block_size=block_size,
    )

    tensors = (
        # input: 设置 other=0，保证 tile 补齐的值不影响 sum
        Tensor(ndim, dtype=dtype, other=0),     
        Tensor(ndim, dtype=dtype),              # output
        Tensor(0, dtype=dtype),                 # kernel_size_flatted (scalar)
    )

    return arrangement_, application, tensors