"""Native Cube selection follows typed SSA and complete affine access proofs."""

import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.backends.emitters.ascendc import AscendCTarget
from ninetoothed.backends.emitters.ascendc_dot import render_cube_dot
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from tests.test_matmul import arrangement


def _matmul(x, y, out):
    accumulator = ntl.zeros(out.shape, dtype=ntl.float32)

    for i in range(x.shape[0]):
        accumulator += ntl.dot(x[i], y[i])

    out = accumulator.to(out.dtype)  # noqa: F841


def _context(
    monkeypatch, shape=(64, 64, 64), *, input_dtype="float16", application=_matmul
):
    contexts = []
    original = AscendCTarget.render_module

    def capture(self, context):
        contexts.append(context)

        return original(self, context)

    monkeypatch.setattr(AscendCTarget, "render_module", capture)
    m, n, k = shape
    DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=functools.partial(
                arrangement, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32
            ),
            application=application,
            tensors=tuple(
                Tensor(shape=matrix_shape, dtype=dtype, other=0)
                for matrix_shape, dtype in (
                    ((m, k), input_dtype),
                    ((k, n), input_dtype),
                    ((m, n), "float16"),
                )
            ),
            backend="ascendc",
        )
    )

    return contexts[-1]


def test_cube_body_preserves_float_accumulation_and_memory_guards(monkeypatch):
    context = _context(monkeypatch)
    rendered = render_cube_dot(context, AscendCTarget(), allow_reassociation=True)
    assert rendered is not None
    assert "Get<float>()" in rendered.body
    assert "mm.cmatrixInitVal = nt_cube_stage == 0 && nt_cube_k == 0" in rendered.body
    assert "Fixpipe<half, float, CFG_ROW_MAJOR>" in rendered.body
    assert "QuantMode_t::F322F16" in rendered.body
    assert "PipeBarrier<PIPE_ALL>()" in rendered.body
    assert "//" not in rendered.body
    assert " <= 65535" in rendered.guard
    assert " % 32 == 0" in rendered.guard
    assert rendered.metadata["ascendc_cube_dot"]["requires_reassociation"]
