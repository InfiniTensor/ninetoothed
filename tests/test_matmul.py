import unittest

import ninetoothed
import torch
from ninetoothed import Symbol, Tensor

from tests.skippers import skip_if_cuda_not_available, skip_if_float8_e5m2_not_supported


def matmul(lhs, rhs):
    BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
    BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)
    BLOCK_SIZE_K = Symbol("BLOCK_SIZE_K", meta=True)

    output_tiled = Tensor(2).tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    lhs_tiled = (
        Tensor(2)
        .tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
        .tile((1, -1))
        .expand((-1, output_tiled.shape[1]))
    )
    rhs_tiled = (
        Tensor(2)
        .tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
        .tile((-1, 1))
        .expand((output_tiled.shape[0], -1))
    )

    @ninetoothed.jit
    def matmul_kernel(lhs: lhs_tiled, rhs: rhs_tiled, output: output_tiled):
        accumulator = ninetoothed.language.zeros(
            output.shape, dtype=ninetoothed.language.float32
        )
        for k in range(lhs.shape[1]):
            accumulator = ninetoothed.language.dot(lhs[0, k], rhs[k, 0], accumulator)
        output = accumulator.to(ninetoothed.language.float16)

    output = torch.empty(
        (lhs.shape[0], rhs.shape[1]), device=lhs.device, dtype=torch.float16
    )

    matmul_kernel(lhs, rhs, output)

    return output


class TestMatMul(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

        shape = (512, 512)

        cls.lhs = torch.randn(shape, device="cuda", dtype=torch.float16)
        cls.rhs = torch.randn(shape, device="cuda", dtype=torch.float16)

    @skip_if_cuda_not_available
    def test_cuda_fp16(self):
        lhs = type(self).lhs
        rhs = type(self).rhs

        self.assertTrue(torch.allclose(matmul(lhs, rhs), torch.matmul(lhs, rhs)))

    @skip_if_cuda_not_available
    @skip_if_float8_e5m2_not_supported
    def test_cuda_fp8(self):
        lhs = type(self).lhs.to(torch.float8_e5m2)
        rhs = type(self).rhs.T.to(torch.float8_e5m2)

        self.assertTrue(
            torch.allclose(
                matmul(lhs, rhs),
                torch.matmul(lhs.to(torch.float16), rhs.to(torch.float16)),
                atol=0.125,
            )
        )


if __name__ == "__main__":
    unittest.main()
