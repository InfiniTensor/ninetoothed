import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

import ntops.kernels.mm as mm


def arrangement(
    input,
    x,
    y,
    beta,
    alpha,
    output,
    input_precision,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    if block_size_m is None:
        block_size_m = mm.BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = mm.BLOCK_SIZE_N

    if block_size_k is None:
        block_size_k = mm.BLOCK_SIZE_K

    _, _, input_arranged, _ = mm.arrangement(
        x,
        y,
        input,
        input_precision,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    x_arranged, y_arranged, output_arranged, _ = mm.arrangement(
        x,
        y,
        output,
        input_precision,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    input_precision_arranged = input_precision

    return (
        input_arranged,
        x_arranged,
        y_arranged,
        beta,
        alpha,
        output_arranged,
        input_precision_arranged,
    )


def application(input, x, y, beta, alpha, output, input_precision):
    mm_output = ntl.zeros(output.shape, dtype=ntl.float32)
    mm.application(x, y, mm_output, input_precision)
    output = beta * input + alpha * mm_output


def premake(
    input_precision=None,
    dtype=None,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    tensors = (
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(2, dtype=dtype),
        Tensor(0, constexpr=True, value=input_precision),
    )

    return arrangement_, application, tensors
