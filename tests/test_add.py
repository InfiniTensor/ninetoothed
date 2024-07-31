import unittest

import ninetoothed
import torch
from ninetoothed import Symbol, Tensor


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


class TestAdd(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available, "CUDA is not available")
    def test_cuda(self):
        torch.manual_seed(0)

        size = 98432

        lhs = torch.rand(size, device="cuda")
        rhs = torch.rand(size, device="cuda")

        self.assertTrue(torch.allclose(add(lhs, rhs), lhs + rhs))


if __name__ == "__main__":
    unittest.main()
