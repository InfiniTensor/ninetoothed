import ninetoothed
import torch
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


class TestAdd:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)

        size = 98432

        cls.lhs = torch.rand(size, device="cuda")
        cls.rhs = torch.rand(size, device="cuda")

    @skip_if_cuda_not_available
    def test_cuda(self):
        lhs = type(self).lhs
        rhs = type(self).rhs

        assert torch.allclose(add(lhs, rhs), lhs + rhs)
