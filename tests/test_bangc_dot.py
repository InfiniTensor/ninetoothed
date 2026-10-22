"""Opt-in native dot is part of normalized options and source emission."""

import functools

import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from tests.test_matmul import arrangement as _arrangement


def _bias(x, y, out):
    value = ntl.zeros(out.shape, dtype=ntl.float32)

    for i in range(x.shape[0]):
        value += ntl.dot(x[i], y[i])

    out = (value + 0.25).to(out.dtype)  # noqa: F841


def _compile():
    return DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=functools.partial(
                _arrangement, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32
            ),
            application=_bias,
            tensors=tuple(
                (nt.Tensor(shape=(64, 64), dtype=nt.float16, other=0) for _ in range(3))
            ),
            backend="bangc",
            backend_options={"task_chunk": 1024},
            max_num_configs=1,
        )
    )


def test_real_arranged_dot_consumes_native_primitive_and_original_epilogue():
    source = _compile().artifact.primary_source
    assert "__bang_matmul(" in source
    assert "__bang_transpose(" in source
    assert "_stride_1) == 1" in source
    assert "} else {" in source
    assert "taskIdX //" not in source
    assert ") //" not in source
    assert "0.25" in source
