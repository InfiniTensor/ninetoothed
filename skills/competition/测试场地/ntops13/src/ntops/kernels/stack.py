import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement(input, output, block_size):
    if block_size is None:
        block_size = ninetoothed.block_size()

    # input: (...,) 任意维度的单个切片
    # output: (...,) 对应的输出切片
    
    # 1. 展平 (Flatten)
    # Stack/Copy 操作是 Element-wise 的，不依赖空间结构，
    # 所以我们可以将数据视为一维连续内存进行处理，以获得最大内存带宽。
    input_arranged = input.flatten()
    output_arranged = output.flatten()
    
    # 2. 切分 (Tile)
    # 将一维数据切分为大小为 block_size 的块
    # 形状变化: (Total_Elements) -> (Num_Blocks, Block_Size)
    input_arranged = input_arranged.tile((block_size,))
    output_arranged = output_arranged.tile((block_size,))

    return input_arranged, output_arranged

def application(input, output):
    # input: (block_size, )
    # output: (block_size, )
    
    # 简单的 Element-wise 赋值
    # DSL 会将其翻译为 load(input) -> store(output)
    output = input

def premake(ndim, block_size=None, dtype=None):
    arrangement_ = functools.partial(
        arrangement,
        block_size=block_size,
    )

    tensors = (
        Tensor(ndim, dtype=dtype),  # input (source tensor)
        Tensor(ndim, dtype=dtype),  # output (destination slice)
    )

    return arrangement_, application, tensors