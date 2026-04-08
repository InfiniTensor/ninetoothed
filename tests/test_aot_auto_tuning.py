import functools
import shutil

import pytest
import torch

import ninetoothed
import ninetoothed.generation
from ninetoothed import Tensor
from tests.utils import get_available_devices


def arrangement(input, other, alpha, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input_arranged = input.tile((block_size,))
    other_arranged = other.tile((block_size,))
    alpha_arranged = alpha
    output_arranged = output.tile((block_size,))

    return input_arranged, other_arranged, alpha_arranged, output_arranged


def application(input, other, alpha, output):
    output = input + alpha * other  # noqa: F841


def premake(size=None, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(shape=(size,), dtype=dtype),
        Tensor(shape=(size,), dtype=dtype),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(shape=(size,), dtype=dtype),
    )

    return arrangement_, application, tensors


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "dtype, ninetoothed_dtype, rtol, atol",
    (
        (torch.float32, ninetoothed.float32, 1e-5, 1e-5),
        (torch.float16, ninetoothed.float16, 1e-3, 1e-3),
    ),
)
@pytest.mark.parametrize("size", (20260128, 1127))
def test_auto_tuning(size, dtype, device, ninetoothed_dtype, rtol, atol):
    caller = device
    kernel_name = "add"
    output_dir = ninetoothed.generation.CACHE_DIR / "test_auto_tuning"

    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir()

    configs = (
        ((), {"size": 20260128, "dtype": ninetoothed.float16, "block_size": 256}, {}),
        ((), {"size": 20260128, "dtype": ninetoothed.float16, "block_size": 1024}, {}),
        ((), {"size": 20260128, "dtype": ninetoothed.float32, "block_size": 512}, {}),
        ((), {"size": 20260128, "dtype": ninetoothed.float32, "block_size": 1024}, {}),
        (
            (),
            {"size": 1127, "dtype": ninetoothed.float16, "block_size": 64},
            {"num_warps": 4},
        ),
        (
            (),
            {"size": 1127, "dtype": ninetoothed.float16, "block_size": 64},
            {"num_warps": 8},
        ),
        (
            (),
            {"size": 1127, "dtype": ninetoothed.float16, "block_size": 256},
            {"num_warps": 4, "num_stages": 1},
        ),
        (
            (),
            {"size": 1127, "dtype": ninetoothed.float16, "block_size": 256},
            {"num_warps": 8, "num_stages": 1},
        ),
        ((), {"size": 1127, "dtype": ninetoothed.float32, "block_size": 512}, {}),
    )

    kernel = ninetoothed.build(
        premake,
        configs,
        meta_parameters=("block_size",),
        caller=caller,
        kernel_name=kernel_name,
        output_dir=output_dir,
    )

    input = torch.randn((size,), dtype=dtype, device=device)
    other = torch.randn((size,), dtype=dtype, device=device)
    alpha = torch.randn((), dtype=torch.float64)
    output = torch.empty_like(input)

    kernel(input, other, alpha, output, size, ninetoothed_dtype)

    shutil.rmtree(output_dir)

    expected = torch.add(input, other, alpha=alpha)

    assert torch.allclose(output, expected, rtol=rtol, atol=atol)
