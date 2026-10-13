import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

import ntops.kernels.mm as mm


def arrangement(
    input,
    mat,
    vec,
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

    # 关键：强制 block_size_n 为 1，因为这是 MV 乘法
    if block_size_n is None:
        block_size_n = 1 

    if block_size_k is None:
        block_size_k = mm.BLOCK_SIZE_K

    # mm.arrangement 现在接收的都是 Rank=2 的 Tensor
    # vec 作为 (K, 1) 矩阵
    # input 作为 (M, 1) 矩阵
    _, _, input_arranged, _ = mm.arrangement(
        mat,
        vec,
        input,
        input_precision,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    mat_arranged, vec_arranged, output_arranged, _ = mm.arrangement(
        mat,
        vec,
        output,
        input_precision,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    input_precision_arranged = input_precision

    return (
        input_arranged,
        mat_arranged,
        vec_arranged,
        beta,
        alpha,
        output_arranged,
        input_precision_arranged,
    )


def application(input, mat, vec, beta, alpha, output, input_precision):
    # 这里 output 是 (M, 1)，zeros 也创建 (M, 1)
    mm_output = ntl.zeros(output.shape, dtype=ntl.float32)
    mm.application(mat, vec, mm_output, input_precision)
    
    # 逐元素操作，形状兼容 (M, 1)
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

    # 修正：将所有 Tensor 定义为 2 维
    # input: (M,) -> (M, 1) Tensor(2)
    # mat:   (M, K) -> Tensor(2)
    # vec:   (K,) -> (K, 1) Tensor(2)
    # output: (M,) -> (M, 1) Tensor(2)
    tensors = (
        Tensor(2, dtype=dtype),  # input (bias) treated as column vector
        Tensor(2, dtype=dtype),  # mat
        Tensor(2, dtype=dtype),  # vec treated as column vector
        Tensor(0, dtype=dtype),  # beta
        Tensor(0, dtype=dtype),  # alpha
        Tensor(2, dtype=dtype),  # output treated as column vector
        Tensor(0, dtype=dtype, constexpr=True, value=input_precision),
    )

    return arrangement_, application, tensors