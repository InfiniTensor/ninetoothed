import torch

import ninetoothed
from ninetoothed import Symbol, Tensor
from tests.skippers import skip_if_cuda_not_available


def add(lhs, rhs):
    BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)

    @ninetoothed.jit
    def add_kernel(
        lhs: Tensor(1).tile((BLOCK_SIZE,)),
        rhs: Tensor(1).tile((BLOCK_SIZE,)),
        output: Tensor(1).tile((BLOCK_SIZE,)),
    ):
        output = lhs + rhs  # noqa: F841

    output = torch.empty_like(lhs)

    add_kernel(lhs, rhs, output)

    return output


@skip_if_cuda_not_available
class TestCUDA:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)

        size = 98432

        cls.lhs = torch.rand(size, device="cuda")
        cls.rhs = torch.rand(size, device="cuda")

    def test_fp32(self):
        lhs = type(self).lhs.to(torch.float32)
        rhs = type(self).rhs.to(torch.float32)

        assert torch.allclose(add(lhs, rhs), lhs + rhs)
