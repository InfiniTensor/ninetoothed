import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement_elementwise(input, other, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()
    
    # 逐元素操作的核心策略：将多维 Tensor 视为展平的 1D 数组
    # 使用 tile 切分数据块
    input = input.flatten().tile((block_size,))
    other = other.flatten().tile((block_size,))
    output = output.flatten().tile((block_size,))
    
    return input, other, output

def application(input, other, output):
    # 调用 DSL 的 maximum 原语
    # 注意：在 Triton/DSL 中，maximum 的 NaN 行为取决于底层实现
    val = ntl.maximum(input, other)
    
    # 生成索引并写回
    indices = ntl.arange(0, input.shape[0])
    output[indices] = val

def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement_elementwise, block_size=block_size)
    tensors = (
        Tensor(ndim, dtype=dtype), # input
        Tensor(ndim, dtype=dtype), # other
        Tensor(ndim, dtype=dtype), # output
    )
    return arrangement_, application, tensors