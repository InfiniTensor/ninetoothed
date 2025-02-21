import random

import torch

import ninetoothed
import tests.test_matmul as matmul
from ninetoothed import Tensor
from tests.skippers import skip_if_cuda_not_available, skip_if_float8_e5m2_not_supported


def arrangement(input, mat1, mat2, beta, alpha, output):
    _, _, input_arranged = matmul.arrangement(mat1, mat2, input)

    mat1_arrange, mat2_arranged, output_arranged = matmul.arrangement(
        mat1, mat2, output
    )

    return input_arranged, mat1_arrange, mat2_arranged, beta, alpha, output_arranged


def application(input, mat1, mat2, beta, alpha, output):
    matmul.application(mat1, mat2, output)
    output = beta * input + alpha * output


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

        # TODO: The current application function inlining feature
        # causes some precision issues. Consider reducing `atol` and
        # `rtol` of this test in the future.
        assert torch.allclose(
            addmm(input, mat1, mat2, beta=beta, alpha=alpha),
            torch.addmm(
                input.to(torch.float16),
                mat1.to(torch.float16),
                mat2.to(torch.float16),
                beta=beta,
                alpha=alpha,
            ),
            atol=0.5,
            rtol=0.5,
        )
