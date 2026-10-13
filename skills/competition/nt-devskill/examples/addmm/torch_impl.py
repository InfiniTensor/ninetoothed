import torch

from examples.addmm.kernel import kernel


def addmm(input, mat1, mat2, beta=1, alpha=1):
    output_shape = (mat1.shape[0], mat2.shape[1])
    output = torch.empty(output_shape, dtype=mat1.dtype, device=mat1.device)

    kernel(input, mat1, mat2, beta, alpha, output)

    return output
