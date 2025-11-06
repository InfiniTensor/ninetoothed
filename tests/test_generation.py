import functools

import pytest

import ninetoothed
import tests.test_matmul as matmul
from ninetoothed import Tensor
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("num_stages", (None, 3, (1, 3)))
@pytest.mark.parametrize("num_warps", (None, 4, (4, 8)))
@pytest.mark.parametrize("block_size_k", (ninetoothed.block_size(), 64))
@pytest.mark.parametrize("block_size_n", (ninetoothed.block_size(), 64))
@pytest.mark.parametrize("block_size_m", (ninetoothed.block_size(), 64))
def test_auto_tuning_generation(
    block_size_m, block_size_n, block_size_k, num_warps, num_stages
):
    auto_tuning_should_be_disabled = (
        isinstance(block_size_m, int)
        and isinstance(block_size_n, int)
        and isinstance(block_size_k, int)
        and not isinstance(num_warps, tuple)
        and not isinstance(num_stages, tuple)
    )

    arrangement = functools.partial(
        matmul.arrangement,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
    )

    application = matmul.application

    tensors = (Tensor(2), Tensor(2), Tensor(2))

    kernel = ninetoothed.make(
        arrangement, application, tensors, num_warps=num_warps, num_stages=num_stages
    )

    source_file = kernel._source

    with open(source_file) as f:
        contents = f.read()

        if auto_tuning_should_be_disabled:
            assert "application_with_auto_tuning" not in contents
            assert "num_warps=" in contents and "num_stages=" in contents
        else:
            assert "application_with_auto_tuning" in contents


@skip_if_cuda_not_available
def test_arrangement_returning_a_single_tensor():
    def arrangement(x):
        return x.tile((ninetoothed.block_size(),))

    def application(x):
        x

    ninetoothed.make(arrangement, application, (Tensor(1),))
