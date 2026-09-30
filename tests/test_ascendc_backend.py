"""Source-level contract tests for the AscendC backend."""

from dataclasses import replace

import pytest

from ninetoothed.backends import Target, emit, normalize_target
from ninetoothed.backends.ascendc import AscendCBackend
from ninetoothed.backends.toolchain import (
    ascendc_compile_command,
    normalize_ascendc_arch,
)
from tests.test_backend_registry import _add_kernel, _matmul_kernel, _source_only_kernel
from tests.utils import requires_backend

pytestmark = requires_backend("ascendc")


class TestAscendCTarget:
    def test_ascendc_target_is_normalized(self):
        assert normalize_target("ascendc") == Target.ASCENDC

    def test_ascendc_options_are_validated(self):
        backend = AscendCBackend()

        assert backend.normalize_options({}) == {"arch": "native"}
        assert backend.normalize_options({"arch": "ascend910b2c"}) == {
            "arch": "dav-c220"
        }
        assert backend.normalize_options({"task_chunk": 512}) == {
            "arch": "native",
            "task_chunk": 512,
        }

        with pytest.raises(ValueError, match="AscendC `arch`"):
            backend.normalize_options({"arch": "sm_90"})

        with pytest.raises(ValueError, match="must be positive"):
            backend.normalize_options({"task_chunk": 0})

    def test_ascendc_arch_aliases_resolve(self):
        assert normalize_ascendc_arch("dav-c220") == "dav-c220"
        assert normalize_ascendc_arch("Ascend910B") == "dav-c220"
        assert normalize_ascendc_arch("910b") == "dav-c220"
        assert normalize_ascendc_arch("native") == "native"

        with pytest.raises(ValueError, match="AscendC `arch`"):
            normalize_ascendc_arch("ascend310p")

    def test_ascendc_compile_command_uses_ccec(self):
        command = ascendc_compile_command(
            "kernel.ascendc",
            "kernel.so",
            arch="ascend910b",
            ccec="/opt/cann/ccec",
        )
        assert command[0] == "/opt/cann/ccec"
        assert "-xcce" in command
        assert "--cce-aicore-arch=dav-c220" in command
        assert "--cce-auto-sync" in command

    def test_ascendc_rejects_source_only_kernel(self):
        with pytest.raises(ValueError, match="requires ssa.Program"):
            emit(_source_only_kernel(), "ascendc")

    def test_ascendc_elementwise_add_uses_block_chunk_domain(self):
        artifact = emit(_add_kernel(), "ascendc")

        assert artifact.language == "ascendc/c++"
        source = artifact.primary_source
        assert 'extern "C" __global__ __aicore__ void add_kernel(' in source
        assert "GM_ADDR x_gm" in source
        assert "nt_gm_x.SetGlobalBuffer((__gm__ float*)x_gm);" in source
        assert "LocalTensor<float> nt_lo" in source
        assert "Add(nt_lo, nt_a0, nt_a1, (int32_t)nt_aligned);" in source
        assert 'extern "C" int launch_add(' in source
        assert "aclrtStream stream" in source
        assert "<<<(uint32_t)(nt_blocks), nullptr, stream>>>" in source
        assert '#include "kernel_operator.h"' in source

    def test_ascendc_task_chunk_option_changes_domain(self):
        kernel = replace(
            _add_kernel(),
            compiler_options={"backend_options": {"task_chunk": 128}},
        )
        artifact = emit(kernel, "ascendc")

        assert "const int64_t nt_chunk = 128;" in artifact.primary_source

    def test_ascendc_matmul_decomposes_into_scalar_loops(self):
        artifact = emit(_matmul_kernel(), "ascendc")
        source = artifact.primary_source

        assert "for (int64_t v10_i = 0; v10_i < k; v10_i += 1)" in source
        assert "linalg.matmul" not in source

    def test_ascendc_bfloat16_is_rejected(self):
        with pytest.raises(ValueError, match="bfloat16"):
            emit(_add_kernel("bfloat16"), "ascendc")

    def test_ascendc_proposes_cooperative_reduction_candidate(self):
        from ninetoothed.backends.ascendc import AscendCOptimizeSchedule
        from ninetoothed.compiler.passes import Context

        candidates = AscendCOptimizeSchedule().schedule_candidates(
            {},
            {
                "granularity": "parallel-reduction",
                "reduction": {"mode": "row-vector", "axis": 1, "extent": 64},
            },
            Context(
                backend=Target.ASCENDC,
                compiler_options={},
                kernel_metadata={},
            ),
        )
        assert tuple(candidate.name for candidate in candidates) == (
            "cooperative-reduction",
        )
        assert candidates[0].schedule == {"ascendc_cooperative_reduction": True}

    def test_ascendc_rejects_collapsed_reduction_stores(self):
        from ninetoothed.frontend.python import from_source
        from ninetoothed.ir import Kernel, TensorSpec

        tensors = (
            TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="input"),
            TensorSpec(ndim=2, shape=("m", 1), dtype="float32", name="output"),
        )
        program = from_source(
            "\nimport ninetoothed.language as ntl\n"
            "def application(input, output):\n"
            "    if input[0, 0] > 0.0:\n"
            "        output = ntl.sum(input, axis=1)\n",
            tensors,
            kind="application",
        )
        kernel = Kernel(
            kernel_name="application",
            source=None,
            tensors=tensors,
            ssa=program,
        )

        with pytest.raises(ValueError, match="cooperative row-vector schedule"):
            emit(kernel, "ascendc")

    def test_ascendc_rejects_atomic_kernels(self):
        from ninetoothed.frontend.python import from_source
        from ninetoothed.ir import Kernel, TensorSpec

        tensors = (
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="input"),
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="output"),
        )
        program = from_source(
            "\nimport ninetoothed.language as ntl\n"
            "def application(input, output):\n"
            "    ntl.atomic_add(output.source.data_ptr(), ntl.sum(input))\n",
            tensors,
            kind="application",
        )
        kernel = Kernel(
            kernel_name="application",
            source=None,
            tensors=tensors,
            ssa=program,
        )

        with pytest.raises(ValueError, match="does not support atomic updates"):
            emit(kernel, "ascendc")

    def test_ascendc_math_support_is_emitted_on_demand(self):
        from ninetoothed.frontend.python import from_source
        from ninetoothed.ir import Kernel, TensorSpec

        tensors = (
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
        )
        program = from_source(
            "\nimport ninetoothed.language as ntl\n"
            "def act(x, out):\n"
            "    out = ntl.exp(x) + ntl.log(x + 1.0)\n",
            tensors,
            kind="act",
        )
        kernel = Kernel(kernel_name="act", source=None, tensors=tensors, ssa=program)
        source = emit(kernel, "ascendc").primary_source

        assert "nt_exp(" in source
        assert "nt_log(" in source
        assert "__aicore__ inline float nt_exp" in source
