"""Bulk staging preserves ordered memory dependencies through whole-domain guards."""

import pytest

import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed.backends.emitters.ascendc import AscendCTarget
from ninetoothed.backends.emitters.bangc import BangCTarget
from ninetoothed.backends.emitters.staging import dense_staging_memory_guard
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest


def _flat(x, out):
    return x.tile((128,)), out.tile((128,))


def _add(x, out):
    out = x + 1.0  # noqa: F841


def _rows(x, out):
    return x.tile((1, x.shape[1])), out.tile((1, out.shape[1]))


def _rms(x, out):
    out = x / ntl.sqrt(ntl.sum(x * x, axis=1) / x.shape[1] + 0.00001)  # noqa: F841


@pytest.mark.parametrize(
    "backend,target", (("bangc", BangCTarget), ("ascendc", AscendCTarget))
)
def test_real_frontend_bulk_routes_consume_the_shared_guard_and_keep_original_fallback(
    monkeypatch, backend, target
):
    row = False
    contexts = []
    render = target.render_module

    def capture(self, context):
        contexts.append(context)

        return render(self, context)

    monkeypatch.setattr(target, "render_module", capture)
    shape = (8, 128) if row else (128,)
    source = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_rows if row else _flat,
            application=_rms if row else _add,
            tensors=tuple(
                (nt.Tensor(shape=shape, dtype=nt.float32, other=0) for _ in range(2))
            ),
            backend=backend,
            backend_options={},
            max_num_configs=1,
        )
    ).artifact.primary_source
    guard = dense_staging_memory_guard(
        contexts[-1],
        ("x",),
        ("out",),
        pointer_suffix="_gm" if backend == "ascendc" else "",
    )
    assert guard != "false" and guard in source
    assert "else {" in source
    assert "__bang_" in source if backend == "bangc" else "TPipe" in source
