import random

import pytest
import torch

import ninetoothed
from ninetoothed import Symbol, Tensor
from tests.utils import skip_if_cuda_not_available

BLOCK_SIZE = Symbol("block_size", constexpr=True)


def arrangement(input, output, block_size=BLOCK_SIZE):
    tile_shape = (1,) + tuple(block_size for _ in range(1, input.ndim))

    return input.tile(tile_shape), output.tile(tile_shape)


def application(input, output):
    output = input  # noqa: F841


def to_padded_tensor(input, padding, jagged_dim, block_size=32):
    max_seq_len = input.offsets().diff().max().item()
    output_shape = tuple(
        size if dim != jagged_dim else max_seq_len
        for dim, size in enumerate(input.shape)
    )

    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    tensors = (
        Tensor(input.ndim, jagged_dim=jagged_dim, other=padding),
        Tensor(output.ndim),
    )

    kernel = ninetoothed.make(arrangement, application, tensors)

    kernel(input, output, block_size=block_size)

    return output


@skip_if_cuda_not_available
@pytest.mark.parametrize("device", ("cuda",))
@pytest.mark.parametrize("padding", (-1,))
@pytest.mark.parametrize("num_batches", (2, 3, 7, 16))
@pytest.mark.parametrize("jagged_dim", (1, 2))
@pytest.mark.parametrize("ndim", (3,))
def test_to_padded_tensor(ndim, jagged_dim, num_batches, padding, device):
    random.seed(0)
    torch.manual_seed(0)

    def _random_size(lower_bound=1, upper_bound=1024):
        return random.randint(lower_bound, upper_bound)

    def _random_batch_shape(batch_ndim):
        return tuple(_random_size() for _ in range(batch_ndim))

    batch_shape = _random_batch_shape(ndim - 1)

    batches = tuple(
        torch.randn(
            batch_shape[: jagged_dim - 1]
            + (_random_size(),)
            + batch_shape[jagged_dim:],
            device=device,
        )
        for _ in range(num_batches)
    )

    input = torch.nested.nested_tensor(batches, layout=torch.jagged)

    output = to_padded_tensor(input, padding=padding, jagged_dim=jagged_dim)

    expected = torch.nested.to_padded_tensor(input, padding)

    assert output.shape == expected.shape and torch.allclose(output, expected)
