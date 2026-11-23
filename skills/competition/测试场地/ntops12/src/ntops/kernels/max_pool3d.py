import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement(input, output, k_d, k_h, k_w, s_d, s_h, s_w, block_size, ceil_mode):
    if block_size is None:
        block_size = ninetoothed.block_size()

    # input: (N, C, D_in, H_in, W_in) 
    # output: (N, C, D_out, H_out, W_out)

    # 1. 窗口切分 (增加 Depth 维度)
    input_arranged = input.tile(
        (1, 1, k_d, k_h, k_w), 
        (1, 1, s_d, s_h, s_w), 
        floor_mode=not ceil_mode
    )
    # => (N, C, D_out, H_out, W_out), dtype=(1, 1, k_d, k_h, k_w)

    # 2. 展平与重排
    input_arranged = input_arranged.ravel()
    # => (N, C, D_out, H_out, W_out, 1, 1, k_d, k_h, k_w)
    
    # 注意：这里 end_dim=5，因为前面有 N,C,D,H,W 5个维度需要合并作为 batch 处理
    input_arranged = input_arranged.flatten(end_dim=5).flatten(start_dim=1)
    # => (N*C*D_out*H_out*W_out, k_d*k_h*k_w)

    # 3. Padding 到最近的 2 的幂次 (用于并行规约)
    # 这里的 padding 值由 premake 中的 other="-inf" 决定
    nearest_pow2 = 1 << (k_d * k_h * k_w - 1).bit_length()
    input_arranged = input_arranged.tile((1, nearest_pow2))
    # => (..., 1), dtype=(1, nearest_pow2)
    
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    input_arranged = input_arranged.tile((block_size, -1))
    input_arranged.dtype = input_arranged.dtype.ravel().squeeze(1)
    # => (..., 1), dtype=(block_size, nearest_pow2)

    # 4. Output 对齐
    output_arranged = output.tile((1, 1, 1, 1, 1)) # (N, C, D, H, W)
    output_arranged = output_arranged.ravel()
    output_arranged = output_arranged.flatten(end_dim=5).flatten(start_dim=1)
    output_arranged = output_arranged.tile((block_size, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)

    return input_arranged, output_arranged

def application(input, output):
    # input: (block_size, nearest_pow2)
    # output: (block_size, )
    
    # Max Pooling 标准操作
    output = ntl.max(input, axis=1)

def premake(ndim, k_d, k_h, k_w, s_d, s_h, s_w, block_size=None, ceil_mode=False, dtype=None):
    arrangement_ = functools.partial(
        arrangement,
        k_d=k_d, k_h=k_h, k_w=k_w,
        s_d=s_d, s_h=s_h, s_w=s_w,
        block_size=block_size,
        ceil_mode=ceil_mode,
    )

    tensors = (
        # input: MaxPool 填充负无穷
        Tensor(ndim, dtype=dtype, other=float("-inf")),     
        Tensor(ndim, dtype=dtype),              # output
    )

    return arrangement_, application, tensors