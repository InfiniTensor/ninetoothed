import functools
import ninetoothed.language as ntl
from ninetoothed import Tensor
# 假设 arrangement 逻辑是通用的，直接复用你例子中的 reduction 模块
from ntops.kernels.reduction import arrangement 

def application(input, output):
    # 均值计算的第一步是求和。
    # 我们在 Kernel 内部做 Accumulation，除法留在外部做以保证数值稳定性。
    accumulator = 0.0
    for i in range(input.shape[0]):
        block_sum = ntl.sum(input[i], axis=0)
        accumulator += block_sum
    # 将累加结果转换为输出类型（通常是 float）
    output[0] = ntl.cast(accumulator, output.dtype.dtype)

def premake(ndim, dim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)
    
    # Mean 算子要求输出必须是浮点数。
    # 如果传入的 dtype 是整数，这里需要根据 DSL 的特性处理，
    # 这里假设 dtype 已经是转换好的 float 类型 (由 Torch 端传入)
    tensors = (
        Tensor(ndim, dtype=dtype), # Input
        Tensor(ndim, dtype=dtype), # Output
    )
    return arrangement_, application, tensors

# --- Global Reduction (All Elements) 的实现 ---

def arrangement_all_elements(input, output, block_size=None):
    input = input.flatten().tile((block_size,))
    output = output.tile((1,))
    return input, output

def application_all_elements(input, output):
    output[0] = ntl.sum(input, 0)

def premake_all_elements(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement_all_elements, block_size=block_size)
    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(1, dtype=dtype),
    )
    return arrangement_, application_all_elements, tensors