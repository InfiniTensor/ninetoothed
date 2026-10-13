import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement(input, output, kernel_size, stride, block_size, ceil_mode):
    if block_size is None:
        block_size = ninetoothed.block_size()

    # input: (N, C, L_in) 
    # output: (N, C, L_out)

    # 1. 窗口切分
    # dim_sizes: (1, 1, kernel_size) -> 在 L 维度上取 kernel_size 长度
    # strides: (1, 1, stride) -> 在 L 维度上步长为 stride
    # floor_mode=not ceil_mode: 决定是否丢弃最后不足一个 kernel 的部分
    input_arranged = input.tile(
        (1, 1, kernel_size), 
        (1, 1, stride), 
        floor_mode=not ceil_mode
    )
    # => (N, C, L_out), dtype=(1, 1, k)

    # 2. 展平与重排
    input_arranged = input_arranged.ravel()
    # => (N, C, L_out, 1, 1, k)
    
    input_arranged = input_arranged.flatten(end_dim=3).flatten(start_dim=1)
    # => (N*C*L_out, k)

    # 3. Padding 到最近的 2 的幂次 (用于并行规约)
    # 这里的 padding 值由 premake 中的 other="-inf" 决定
    nearest_pow2 = 1 << (kernel_size - 1).bit_length()
    input_arranged = input_arranged.tile((1, nearest_pow2))
    # => (..., 1), dtype=(1, nearest_pow2)
    
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    input_arranged = input_arranged.tile((block_size, -1))
    input_arranged.dtype = input_arranged.dtype.ravel().squeeze(1)
    # => (..., 1), dtype=(block_size, nearest_pow2)

    # 4. Output 对齐
    output_arranged = output.tile((1, 1, 1)) # (N, C, L_out)
    output_arranged = output_arranged.ravel()
    output_arranged = output_arranged.flatten(end_dim=3).flatten(start_dim=1)
    output_arranged = output_arranged.tile((block_size, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)

    return input_arranged, output_arranged

def application(input, output):
    # input: (block_size, nearest_pow2)
    # output: (block_size, )
    
    # 直接取 Max，padding 值为 -inf，不影响结果
    output = ntl.max(input, axis=1)

def premake(ndim, kernel_size, stride, block_size=None, ceil_mode=False, dtype=None):
    arrangement_ = functools.partial(
        arrangement,
        kernel_size=kernel_size,
        stride=stride,
        block_size=block_size,
        ceil_mode=ceil_mode,
    )

    tensors = (
        # input: MaxPool 填充负无穷
        Tensor(ndim, dtype=dtype, other=float("-inf")),     
        Tensor(ndim, dtype=dtype),              # output
    )

    return arrangement_, application, tensors