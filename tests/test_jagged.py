import functools
import random

import pytest
import torch

import ninetoothed
from ninetoothed import Symbol, Tensor
from tests.utils import get_available_devices


def _nested_tensor_from_batches(batches, jagged_dim):
    # Explicit packing supports both jagged dimensions on PyTorch 2.5.
    packed_dim = jagged_dim - 1
    offsets = [0]

    for batch in batches:
        offsets.append(offsets[-1] + batch.shape[packed_dim])

    values = torch.cat(batches, dim=packed_dim)

    return torch.nested.nested_tensor_from_jagged(
        values,
        torch.tensor(offsets, dtype=torch.int64, device=values.device),
        jagged_dim=jagged_dim,
    )


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

    input = _nested_tensor_from_batches(batches, jagged_dim)
    expected_shape = (num_batches,) + tuple(
        max(batch.shape[dim] for batch in batches) for dim in range(ndim - 1)
    )
    expected = torch.full(
        expected_shape, padding, dtype=batches[0].dtype, device=device
    )

    for index, batch in enumerate(batches):
        expected[(index,) + tuple(slice(0, size) for size in batch.shape)] = batch

    output = to_padded_tensor(input, padding=padding, jagged_dim=jagged_dim)

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

    dst = _nested_tensor_from_batches(batches, jagged_dim)
    src = torch.randn(
        tuple(size if dim != jagged_dim else 1 for dim, size in enumerate(dst.shape)),
        dtype=dst.dtype,
        device=dst.device,
    )

    expected = torch.cat(
        tuple(src[index].expand(batch.shape) for index, batch in enumerate(batches)),
        dim=jagged_dim - 1,
    )
    expected_offsets = dst.offsets().clone()

    copy(dst, src, jagged_dim)

    actual = dst.values()
    assert actual.shape == expected.shape and torch.allclose(actual, expected)
    assert torch.equal(dst.offsets(), expected_offsets)
