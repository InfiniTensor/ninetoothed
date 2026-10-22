"""Actual affine SSA retains exact source spans and guarded output ownership."""

import ninetoothed as nt
from ninetoothed.backends.ascendc_resources import ASCENDC_UB_BYTES
from ninetoothed.backends.emitters.ascendc import AscendCTarget
from ninetoothed.backends.emitters.ascendc_affine import render_affine_transfer
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest


def _crop(x, out):
    return x[1:-1].tile((256,)), out.tile((256,))


def _copy(x, out):
    out = x  # noqa: F841


def _context(
    monkeypatch,
    arrangement,
    sizes,
):
    contexts = []
    original = AscendCTarget.render_module

    def capture(self, context):
        contexts.append(context)

        return original(self, context)

    monkeypatch.setattr(AscendCTarget, "render_module", capture)
    DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=_copy,
            tensors=tuple(
                nt.Tensor(shape=(size,), dtype=nt.float32, other=0) for size in sizes
            ),
            backend="ascendc",
            max_num_configs=1,
        )
    )

    return contexts[-1]


def test_real_transfer_uses_exact_byte_counts_and_bounded_local_gather(monkeypatch):
    arrangement, sizes = (_crop, (1009, 1007))
    plan = render_affine_transfer(_context(monkeypatch, arrangement, sizes))
    assert plan is not None
    assert "(nt_affine_count - 1)" in plan.body
    assert "nt_affine_span * sizeof(float)" in plan.body
    assert "nt_affine_count * sizeof(float)" in plan.body
    assert "DataCopyPadExtParams<float>" in plan.body
    assert "Gather(nt_affine_values" in plan.body
    assert "ReinterpretCast<uint32_t>()" in plan.body
    assert "* 4LL)" in plan.body
    assert "sizeof(float)" not in plan.body.split("Muls(")[1].split(";")[0]
    assert "31ULL" in plan.guard and "<= 4" in plan.guard
    assert plan.ub_bytes <= ASCENDC_UB_BYTES
    assert plan.chunk % 8 == 0

    assert "!= 1" not in plan.guard
