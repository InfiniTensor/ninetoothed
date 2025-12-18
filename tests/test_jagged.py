import functools
import random

import pytest
import torch

import ninetoothed
from ninetoothed import Symbol, Tensor
from tests.utils import get_available_devices


class ToPaddedTensor:
    BLOCK_SIZE = Symbol("block_size", constexpr=True)

    @staticmethod
    def arrangement(input, output, block_size=BLOCK_SIZE):
        tile_shape = (1,) + tuple(block_size for _ in range(1, input.ndim))

        return input.tile(tile_shape), output.tile(tile_shape)

    @staticmethod
    def application(input, output):
        output = input  # noqa: F841

    @staticmethod
    def premake(ndim, padding, jagged_dim, block_size=None):
        if block_size is not None:
            arrangement = functools.partial(
                ToPaddedTensor.arrangement, block_size=block_size
            )
        else:
            arrangement = ToPaddedTensor.arrangement

        tensors = (Tensor(ndim, jagged_dim=jagged_dim, other=padding), Tensor(ndim))

        return arrangement, ToPaddedTensor.application, tensors


def to_padded_tensor(input, padding, jagged_dim, block_size=32):
    max_seq_len = input.offsets().diff().max().item()
    output_shape = tuple(
        size if dim != jagged_dim else max_seq_len
        for dim, size in enumerate(input.shape)
    )

    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    kernel = ninetoothed.make(*ToPaddedTensor.premake(input.ndim, padding, jagged_dim))

    kernel(input, output, block_size=block_size)

    return output


@pytest.mark.parametrize("device", get_available_devices())
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
