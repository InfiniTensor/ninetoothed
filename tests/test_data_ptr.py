import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from tests.utils import skip_if_cuda_not_available


def arrangement(input, output, BLOCK_SIZE=1024):
    input_arranged = input.tile((BLOCK_SIZE,))
    output_arranged = output.tile((-1,)).expand(input_arranged.shape)

    return input_arranged, output_arranged


def application(input, output):
    ntl.atomic_add(output.data_ptr(), ntl.sum(input))


def sum(input):
    output = torch.empty(1, dtype=input.dtype, device=input.device)

    tensors = (Tensor(1), Tensor(1, shape_options={"constexpr": True}))

    sum_kernel = ninetoothed.make(arrangement, application, tensors)

    sum_kernel(input, output)

    return output[0]


@skip_if_cuda_not_available
def test_data_ptr():
    torch.manual_seed(0)

    size = 64180

    input = torch.randn(size, device="cuda")

    output = sum(input)

    expected = torch.sum(input)

    assert output.shape == expected.shape and torch.allclose(output, expected)
