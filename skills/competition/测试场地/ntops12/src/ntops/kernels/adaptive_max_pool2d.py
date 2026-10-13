import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement(input, output, kernel_size_h, kernel_size_w, stride_h, stride_w, block_size):
    if block_size is None:
        block_size = ninetoothed.block_size()

    # input: (N, C, H_in, W_in) 
    # output: (N, C, H_out, W_out)
    
    # 使用 tile 将输入切分为窗口
    # floor_mode=True 对应默认行为，对于 Adaptive Pool，我们通常通过计算好的 stride/kernel 确保覆盖
    input_arranged = input.tile(
        (1, 1, kernel_size_h, kernel_size_w), 
        (1, 1, stride_h, stride_w)
    )
    # => (N, C, H_out, W_out), dtype=(1, 1, k_h, k_w)
    
    input_arranged = input_arranged.ravel()
    # => (N, C, H_out, W_out, 1, 1, k_h, k_w)
    
    input_arranged = input_arranged.flatten(end_dim=4).flatten(start_dim=1)
    # => (N*C*H_out*W_out, k_h*k_w)

    # 找到最近的 2 的倍数用于并行规约
    nearest_pow2 = 1 << (kernel_size_h * kernel_size_w - 1).bit_length()
    input_arranged = input_arranged.tile((1, nearest_pow2))
    # => (..., 1), dtype=(1, nearest_pow2)
    
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    # => (..., 1), dtype=(nearest_pow2, )
    
    input_arranged = input_arranged.tile((block_size, -1))
    # => (..., 1), dtype=(block_size, nearest_pow2)
    input_arranged.dtype = input_arranged.dtype.ravel().squeeze(1)

    # 处理 output 的 layout 以匹配 input 的 block_size
    output_arranged = output.tile((1, 1, 1, 1))
    output_arranged = output_arranged.ravel()
    output_arranged = output_arranged.flatten(end_dim=4).flatten(start_dim=1)
    output_arranged = output_arranged.tile((block_size, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)

    return input_arranged, output_arranged

def application(input, output):
    # input: (block_size, nearest_pow2) 
    # output: (block_size, )
    
    # 简单的 max reduction
    # 因为在 premake 中设置了 other=float("-inf")，padding 部分的值为负无穷，
    # 或者是 nearest_pow2 补齐产生的部分，通常默认为 0 或 padding 值，
    # 这里为了安全，可以显式处理 padding，但如果 arrange padding 正确，直接 max 即可。
    # 假设 DSL 的 tile 填充行为遵循 Tensor 的 other 属性。
    
    output = ntl.max(input, axis=1)

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
        # input: 设置 other 为负无穷，这样 tile 越界填充的值不会影响 max
        Tensor(ndim, dtype=dtype, other=float("-inf")), 
        Tensor(ndim, dtype=dtype),              # output
    )

    return arrangement_, application, tensors