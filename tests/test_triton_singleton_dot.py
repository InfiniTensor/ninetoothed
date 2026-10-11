from dataclasses import replace

import pytest
import torch

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.backends.triton_singleton_dot import TritonSingletonDot
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.compiler.passes import Context, default_spec
from tests.utils import get_available_devices

PASS = "ssa.triton.singleton_dot"


def _arrangement(a, b, out):
    return a.tile((-1, -1)), b.tile((-1, -1)), out.tile((-1, -1))


def _application(a, b, out):
    out = ntl.dot(a, b)  # noqa: F841


def _request(m=1, k=32, n=16, dtype="float16"):
    passes = list(default_spec("triton").passes)
    passes.insert(passes.index("ssa.decompose_linalg"), PASS)

    return CompileRequest(
        arrangement=_arrangement,
        application=_application,
        tensors=(
            Tensor(shape=(m, k), dtype=dtype),
            Tensor(shape=(k, n), dtype=dtype),
            Tensor(shape=(m, n), dtype="float32"),
        ),
        backend="triton",
        pipeline=tuple(passes),
        num_warps=4,
        max_num_configs=1,
    )


def test_singleton_dot_legality():
    compilation = DEFAULT_COMPILER.compile(_request())
    context = Context(
        backend=compilation.target.backend,
        compiler_options={},
        kernel_metadata={},
        tensors=compilation.kernel.tensors,
    )
    transform = TritonSingletonDot()
    program = compilation.kernel.ssa
    assert transform.run(program, context) != program
    limited = replace(
        context,
        pass_options={"ssa.triton.block_reductions": {"max_block_elements": 16}},
    )
    assert transform.run(program, limited) == program

    original = DEFAULT_COMPILER.compile(replace(_request(m=16), pipeline=None)).kernel
    assert (
        transform.run(original.ssa, replace(context, tensors=original.tensors))
        == original.ssa
    )

    integer_inputs = replace(
        program,
        inputs=tuple(
            replace(value, type=replace(value.type, dtype="int32"))
            for value in program.inputs
        ),
    )
    assert transform.run(integer_inputs, context) == integer_inputs
    assert PASS not in default_spec("triton").passes


@pytest.mark.parametrize("device", get_available_devices())
def test_singleton_dot_runtime(device):
    for m, n in ((1, 16), (16, 1)):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            a = torch.randn(m, 64, device=device, dtype=dtype)[:, ::2]
            b = torch.randn(32, n, device=device, dtype=dtype)
            out = torch.empty((m, n), device=device, dtype=torch.float32)
            compilation = DEFAULT_COMPILER.compile(
                _request(m=m, n=n, dtype=str(dtype).split(".")[-1])
            )
            assert "tl.dot(" not in compilation.artifact.primary_source
            assert "tl.sum(" in compilation.artifact.primary_source
            DEFAULT_COMPILER.materialize(compilation)(a, b, out)
            torch.testing.assert_close(
                out, (a.double() @ b.double()).float(), rtol=2e-5, atol=2e-5
            )
