import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor
from tests.skippers import skip_if_cuda_not_available, skip_if_float8_e5m2_not_supported

BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)
BLOCK_SIZE_K = Symbol("BLOCK_SIZE_K", meta=True)


def arrangement(
    lhs,
    rhs,
    output,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    output_tiled = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    lhs_tiled = (
        lhs.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
        .tile((1, -1))
        .expand((-1, output_tiled.shape[1]))
    )
    lhs_tiled.dtype = lhs_tiled.dtype.squeeze(0)

    rhs_tiled = (
        rhs.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
        .tile((-1, 1))
        .expand((output_tiled.shape[0], -1))
    )
    rhs_tiled.dtype = rhs_tiled.dtype.squeeze(1)

    return lhs_tiled, rhs_tiled, output_tiled


def application(lhs, rhs, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(lhs.shape[0]):
        accumulator += ntl.dot(lhs[k], rhs[k])
    output = accumulator.to(ntl.float16)


def matmul(lhs, rhs):
    output = torch.empty(
        (lhs.shape[0], rhs.shape[1]), device=lhs.device, dtype=torch.float16
    )

    matmul_kernel = ninetoothed.make(
        arrangement, application, (Tensor(2), Tensor(2), Tensor(2))
    )

    matmul_kernel(lhs, rhs, output)

    return output


@skip_if_cuda_not_available
class TestCUDA:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)

        shape = (512, 512)

        cls.lhs = torch.randn(shape, device="cuda")
        cls.rhs = torch.randn(shape, device="cuda")

    def test_fp16(self):
        lhs = type(self).lhs.to(torch.float16)
        rhs = type(self).rhs.to(torch.float16)

        assert torch.allclose(matmul(lhs, rhs), torch.matmul(lhs, rhs))

    @skip_if_float8_e5m2_not_supported
    def test_fp8(self):
        lhs = type(self).lhs.to(torch.float8_e5m2)
        rhs = type(self).rhs.T.to(torch.float8_e5m2)

        assert torch.allclose(
            matmul(lhs, rhs),
            torch.matmul(lhs.to(torch.float16), rhs.to(torch.float16)),
            atol=0.125,
        )
