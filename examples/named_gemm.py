"""Complete S1 GEMM definition: arrangement, computation, and compilation."""

import ninetoothed as nt
import ninetoothed.language as ntl

BM, BN, BK = 16, 64, 32


def arrangement(lhs, rhs, output):
    return (
        lhs.axes("m", "k").tile((BM, BK)),
        rhs.axes("k", "n").tile((BK, BN)),
        output.axes("m", "n").tile((BM, BN)),
    )


def application(lhs, rhs, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(lhs.shape[0]):
        accumulator = accumulator + ntl.dot(lhs[k], rhs[k])

    output = accumulator.to(output.dtype)  # noqa: F841


def make(dtype="float16"):
    """Compile the definition without choosing concrete logical M/N/K sizes."""
    return nt.make(
        arrangement,
        application,
        tuple(nt.Tensor(2, dtype=dtype) for _ in range(3)),
    )
