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


def _nested_tensor_or_skip(batches):
    try:
        return torch.nested.nested_tensor(batches, layout=torch.jagged)
    except NotImplementedError as error:
        pytest.skip(f"jagged nested tensors are unavailable: {error}")
    except RuntimeError as error:
        if "Cannot represent given tensor list" not in str(error):
            raise

        pytest.skip(f"requested jagged dimension is unavailable: {error}")


def _manual_padded_reference(batches, output, padding, jagged_dim):
    expected = torch.full_like(output, padding)

    for batch_index, batch in enumerate(batches):
        index = [batch_index, *(slice(None) for _ in range(output.ndim - 1))]
        index[jagged_dim] = slice(0, batch.shape[jagged_dim - 1])
        expected[tuple(index)] = batch
    return expected


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("padding", (-1,))
@pytest.mark.parametrize("num_batches", (2, 3, 7, 16))
@pytest.mark.parametrize("jagged_dim", (1, 2))
@pytest.mark.parametrize("ndim", (3,))
def test_to_padded_tensor(ndim, jagged_dim, num_batches, padding, device):
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

    input = _nested_tensor_or_skip(batches)

    output = to_padded_tensor(input, padding=padding, jagged_dim=jagged_dim)

    expected = _manual_padded_reference(batches, output, padding, jagged_dim)

    assert output.shape == expected.shape and torch.allclose(output, expected)


class Copy:
    BLOCK_SIZE = Symbol("block_size", constexpr=True)

    @staticmethod
    def arrangement(dst, src, jagged_dim, block_size=BLOCK_SIZE):
        tile_shape = (1,) + tuple(block_size for _ in range(1, dst.ndim))

        return dst.tile(tile_shape), src.expand(
            tuple(
                -1 if dim != jagged_dim else dst.shape[dim] for dim in range(src.ndim)
            )
        ).tile(tile_shape)

    @staticmethod
    def application(dst, src):
        dst = src  # noqa: F841

    @staticmethod
    def premake(ndim, jagged_dim, block_size=None):
        arrangement = functools.partial(Copy.arrangement, jagged_dim=jagged_dim)

        if block_size is not None:
            arrangement = functools.partial(Copy.arrangement, block_size=block_size)

        tensors = (Tensor(ndim, jagged_dim=jagged_dim), Tensor(ndim))

        return arrangement, Copy.application, tensors


def copy(dst, src, jagged_dim, block_size=32):
    kernel = ninetoothed.make(*Copy.premake(dst.ndim, jagged_dim))

    kernel(dst, src, block_size=block_size)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("num_batches", (2, 3, 7, 16))
@pytest.mark.parametrize("jagged_dim", (1, 2))
@pytest.mark.parametrize("ndim", (3,))
def test_expand(ndim, jagged_dim, num_batches, device):
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

    dst = _nested_tensor_or_skip(batches)
    src = torch.randn(
        tuple(size if dim != jagged_dim else 1 for dim, size in enumerate(dst.shape)),
        dtype=dst.dtype,
        device=dst.device,
    )

    copy(dst, src, jagged_dim)

    for batch_index, actual in enumerate(dst.unbind()):
        expected = src[batch_index].expand_as(actual)
        assert actual.shape == expected.shape and torch.allclose(actual, expected)
