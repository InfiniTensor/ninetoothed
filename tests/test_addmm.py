import random

import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
import tests.test_matmul as matmul
from ninetoothed import Tensor
from tests.utils import get_available_devices


def arrangement(
    input,
    mat1,
    mat2,
    beta,
    alpha,
    output,
    BLOCK_SIZE_M=matmul.BLOCK_SIZE_M,
    BLOCK_SIZE_N=matmul.BLOCK_SIZE_N,
    BLOCK_SIZE_K=matmul.BLOCK_SIZE_K,
):
    _, _, input_arranged = matmul.arrangement(
        mat1, mat2, input, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

    mat1_arranged, mat2_arranged, output_arranged = matmul.arrangement(
        mat1, mat2, output, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

    return input_arranged, mat1_arranged, mat2_arranged, beta, alpha, output_arranged


def application(input, mat1, mat2, beta, alpha, output):
    matmul_output = ntl.zeros(output.shape, dtype=ntl.float32)
    matmul.application(mat1, mat2, matmul_output)
    output = beta * input + alpha * matmul_output


def addmm(input, mat1, mat2, beta=1, alpha=1):
    output = torch.empty(
        (mat1.shape[0], mat2.shape[1]), device=mat1.device, dtype=torch.float16
    )

    addmm_kernel = ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(2), Tensor(2), Tensor(0), Tensor(0), Tensor(2)),
    )

    addmm_kernel(input, mat1, mat2, beta, alpha, output)

    return output


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "dtype, atol", ((torch.float16, 0.075),) + matmul._FLOAT8_E5M2_CONFIG
)
@pytest.mark.parametrize("k", (512,))
@pytest.mark.parametrize("n", (512,))
@pytest.mark.parametrize("m", (512,))
def test(m, n, k, dtype, device, atol):
    random.seed(0)
    torch.manual_seed(0)

    randn_dtype = dtype if dtype != torch.float8_e5m2 else torch.float16

    input = torch.randn((m, n), dtype=randn_dtype, device=device)
    mat1 = torch.randn((m, k), dtype=randn_dtype, device=device)
    mat2 = torch.randn((k, n), dtype=randn_dtype, device=device)
    beta = random.uniform(0, 1)
    alpha = random.uniform(0, 1)

    if dtype == torch.float8_e5m2:
        input = input.to(dtype)
        mat1 = mat1.to(dtype)
        mat2 = mat2.T.to(dtype)

        output = addmm(input, mat1, mat2, beta=beta, alpha=alpha)
        expected = torch.addmm(
            input.to(torch.float16),
            mat1.to(torch.float16),
            mat2.to(torch.float16),
            beta=beta,
            alpha=alpha,
        )
    else:
        output = addmm(input, mat1, mat2, beta=beta, alpha=alpha)
        expected = torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

    assert torch.allclose(output, expected, atol=atol)
