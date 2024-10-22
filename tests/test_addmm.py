import random

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor
from tests.skippers import skip_if_cuda_not_available, skip_if_float8_e5m2_not_supported


def addmm(input, mat1, mat2, beta=1, alpha=1):
    BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
    BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)
    BLOCK_SIZE_K = Symbol("BLOCK_SIZE_K", meta=True)

    input_tiled = Tensor(2).tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    output_tiled = Tensor(2).tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    mat1_tiled = (
        Tensor(2)
        .tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
        .tile((1, -1))
        .expand((-1, output_tiled.shape[1]))
    )
    mat1_tiled.dtype = mat1_tiled.dtype.squeeze(0)

    mat2_tiled = (
        Tensor(2)
        .tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
        .tile((-1, 1))
        .expand((output_tiled.shape[0], -1))
    )
    mat2_tiled.dtype = mat2_tiled.dtype.squeeze(1)

    @ninetoothed.jit
    def addmm_kernel(
        input: input_tiled,
        mat1: mat1_tiled,
        mat2: mat2_tiled,
        beta: Tensor(0),
        alpha: Tensor(0),
        output: output_tiled,
    ):
        accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
        for k in range(mat1.shape[0]):
            accumulator += ntl.dot(mat1[k], mat2[k])
        output = beta * input + alpha * accumulator.to(ntl.float16)

    output = torch.empty(
        (mat1.shape[0], mat2.shape[1]), device=mat1.device, dtype=torch.float16
    )

    addmm_kernel(input, mat1, mat2, beta, alpha, output)

    return output


@skip_if_cuda_not_available
class TestCUDA:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)

        shape = (512, 512)

        cls.input = torch.randn(shape, device="cuda")
        cls.mat1 = torch.randn(shape, device="cuda")
        cls.mat2 = torch.randn(shape, device="cuda")
        cls.beta = random.uniform(0, 1)
        cls.alpha = random.uniform(0, 1)

    def test_fp16(self):
        input = type(self).input.to(torch.float16)
        mat1 = type(self).mat1.to(torch.float16)
        mat2 = type(self).mat2.to(torch.float16)
        beta = type(self).beta
        alpha = type(self).alpha

        assert torch.allclose(
            addmm(input, mat1, mat2, beta=beta, alpha=alpha),
            torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha),
            atol=0.075,
        )

    @skip_if_float8_e5m2_not_supported
    def test_fp8(self):
        input = type(self).input.to(torch.float8_e5m2)
        mat1 = type(self).mat1.to(torch.float8_e5m2)
        mat2 = type(self).mat2.T.to(torch.float8_e5m2)
        beta = type(self).beta
        alpha = type(self).alpha

        assert torch.allclose(
            addmm(input, mat1, mat2, beta=beta, alpha=alpha),
            torch.addmm(
                input.to(torch.float16),
                mat1.to(torch.float16),
                mat2.to(torch.float16),
                beta=beta,
                alpha=alpha,
            ),
            atol=0.125,
        )
