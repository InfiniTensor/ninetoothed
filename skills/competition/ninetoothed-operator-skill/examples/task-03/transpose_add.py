import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def _arrangement(input, bias, output):
    block_shape = (ninetoothed.block_size(), ninetoothed.block_size())
    return input.tile(block_shape), bias.tile(block_shape), output.tile(block_shape)


def _application(input_t, bias, output):
    values = ntl.load(
        input_t.source.data_ptr()
        + input_t.offsets(0)[:, None] * input_t.source.stride(0)
        + input_t.offsets(1)[None, :] * input_t.source.stride(1)
    )
    output = values + bias  # noqa: F841


_KERNEL = ninetoothed.make(
    _arrangement, _application, (Tensor(2), Tensor(2), Tensor(2))
)


def transpose_add(input, bias):
    """Return input.transpose(0, 1) + bias while respecting input strides."""
    if input.dim() != 2 or bias.dim() != 2:
        raise NotImplementedError(
            f"transpose_add expects 2D tensors, "
            f"got input={tuple(input.shape)}, bias={tuple(bias.shape)}"
        )

    output_shape = (input.shape[1], input.shape[0])

    if tuple(bias.shape) != output_shape:
        raise ValueError(f"bias shape must be {output_shape}, got {tuple(bias.shape)}")

    # Use PyTorch transpose to create a view with transposed shape
    # but original stride info. This avoids ninetoothed's permute
    # which causes dimension mismatch in compilation.
    transposed_input = input.transpose(0, 1)

    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    _KERNEL(transposed_input, bias, output)
    return output
