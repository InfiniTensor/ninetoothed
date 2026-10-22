"""Source-level contract tests for the AscendC backend."""

import pytest

from ninetoothed.backends import emit
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import Kernel, TensorSpec


def _emit_application(name, source, tensors):
    program = from_source(source, tensors, kind=name)

    return emit(
        Kernel(kernel_name=name, source=None, tensors=tensors, ssa=program), "ascendc"
    )


class TestAscendCTarget:
    def test_fused_vector_launch_covers_the_selected_ub_chunk(self):
        tensors = tuple(
            TensorSpec(ndim=1, shape=(16385,), dtype="float32", name=name)
            for name in ("x", "y", "out")
        )

        application_source = "def fused(x, y, out):\n    out = (x + y) * 2.0\n"

        source = _emit_application("fused", application_source, tensors).primary_source

        assert "nt_launch_logical * 4096" in source
        assert "nt_launch_total = ((" in source
        assert "(16385 + 4096 - 1) / 4096" in source
        assert "(((uintptr_t)x_gm) & 31ULL) == 0" in source
        assert "(((uintptr_t)out_gm) & 31ULL) == 0" in source
        assert "if (nt_launch_logical != 0)" in source
        assert "for (int64_t index = 0; index < 16385; index++)" in source

    @pytest.mark.parametrize("expression", ("x / y", "ntl.rsqrt(x)"))
    def test_strict_vector_estimates_refine_and_preserve_exceptional_lanes(
        self, expression
    ):
        tensors = tuple(
            (
                TensorSpec(ndim=1, shape=(8193,), dtype="float32", name=name)
                for name in ("x", "y", "out")
            )
        )
        application_source = f"import ninetoothed.language as ntl\ndef precise(x, y, out):\n    out = {expression}\n"
        source = _emit_application(
            "precise", application_source, tensors
        ).primary_source
        assert "nt_iteration < 2" in source
        assert "CompareScalar(nt_ref_mask0" in source
        assert "nt_valid >> nt_lane" in source
        assert "HardEvent::V_S" in source
        assert "HardEvent::S_V" in source
