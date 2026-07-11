import torch

import ninetoothed
from ninetoothed import Symbol, Tensor, block_size

BLOCK_SIZE = block_size()
BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()

_KERNELS = {}


def _add_1d(lhs, rhs, output):
    block = Symbol("BLOCK_SIZE", meta=True)

    @ninetoothed.jit
    def add_kernel(
        lhs: Tensor(1).tile((block,)),
        rhs: Tensor(1).tile((block,)),
        output: Tensor(1).tile((block,)),
    ):
        output = lhs + rhs  # noqa: F841

    add_kernel(lhs, rhs, output)


def _align_to_output(tensor, output_shape):
    """Align tensor to output shape via broadcast, returning a torch.Tensor."""
    aligned = tensor

    while aligned.ndim < len(output_shape):
        aligned = aligned.unsqueeze(0)

    # Use expand to broadcast to output shape
    aligned = aligned.expand(output_shape)

    return aligned


def _arrangement_2d(
    lhs, rhs, output, block_size_m=BLOCK_SIZE_M, block_size_n=BLOCK_SIZE_N
):
    block = (block_size_m, block_size_n)

    output_arranged = output.tile(block)
    lhs_arranged = lhs.tile(block)
    rhs_arranged = rhs.tile(block)

    return lhs_arranged, rhs_arranged, output_arranged


def _application(lhs, rhs, output):
    output = lhs + rhs  # noqa: F841


def _get_kernel(key, arrangement, tensors):
    if key not in _KERNELS:
        _KERNELS[key] = ninetoothed.make(arrangement, _application, tensors)

    return _KERNELS[key]


def add(lhs, rhs):
    output_shape = torch.broadcast_shapes(lhs.shape, rhs.shape)
    output = torch.empty(output_shape, dtype=lhs.dtype, device=lhs.device)

    if lhs.dim() == 1 and rhs.dim() == 1:
        _add_1d(lhs, rhs, output)
    elif len(output_shape) == 2:
        # Align lhs and rhs to output shape before passing to kernel
        lhs_aligned = _align_to_output(lhs, output_shape)
        rhs_aligned = _align_to_output(rhs, output_shape)

        key = ("2d", lhs_aligned.dim(), rhs_aligned.dim())
        tensors = (Tensor(2), Tensor(2), Tensor(2))
        kernel = _get_kernel(key, _arrangement_2d, tensors)
        kernel(lhs_aligned, rhs_aligned, output)
    else:
        raise NotImplementedError(
            f"Unsupported broadcast pattern: "
            f"lhs.shape={lhs.shape}, rhs.shape={rhs.shape}"
        )

    return output
