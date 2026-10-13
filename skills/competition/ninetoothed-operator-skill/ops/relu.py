"""
Operator: relu
Pattern: Elementwise (generic ndim via flatten + autotuning block_size)
Formula: output = max(input, 0)
Reference: torch.relu(input)

Calling convention:
    kernel = make_relu(ndim=1)
    kernel(input, output)   # NO BLOCK_SIZE kwarg — autotuning handles it
"""
import ninetoothed
from ninetoothed import Tensor


def _arrangement(*tensors):
    block_size = ninetoothed.block_size()
    ndim = max(t.ndim for t in tensors)
    return tuple(
        t.flatten().tile((block_size,)) if t.ndim != 0 else t
        for t in tensors
    )


def _application(input, output):
    output = max(0.0, input)  # noqa: F841


def make_relu(ndim: int = 1):
    tensors = (Tensor(ndim), Tensor(ndim))
    return ninetoothed.make(_arrangement, _application, tensors)


# Default 1D kernel
kernel = make_relu(ndim=1)
