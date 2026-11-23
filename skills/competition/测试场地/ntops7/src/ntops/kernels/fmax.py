import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement_elementwise(input, other, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()
    
    input = input.flatten().tile((block_size,))
    other = other.flatten().tile((block_size,))
    output = output.flatten().tile((block_size,))
    
    return input, other, output

def application(input, other, output):
    # Compute element-wise maximum
    val = ntl.maximum(input, other)
    
    # Generate index range [0, block_size) to match the shape of 'val'
    # input.shape[0] retrieves the block_size of the tile
    indices = ntl.arange(0, input.shape[0])
    
    # Write the vector 'val' to the memory locations defined by output + indices
    output[indices] = val

def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement_elementwise, block_size=block_size)
    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )
    return arrangement_, application, tensors