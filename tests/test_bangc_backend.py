"""Source-level contract tests for the BangC backend."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ninetoothed.backends import Target, emit, normalize_target
from ninetoothed.backends.bangc import BangCBackend
from ninetoothed.backends.toolchain import (
    bangc_compile_command,
    normalize_bangc_arch,
)
from tests.test_backend_registry import _add_kernel, _matmul_kernel, _source_only_kernel
from tests.utils import requires_backend

pytestmark = requires_backend("bangc")


class TestBangCTarget:
    def test_bang_target_is_normalized(self):
        assert normalize_target("bangc") == Target.BANGC

    def test_bangc_options_are_validated(self):
        backend = BangCBackend()

        assert backend.normalize_options({}) == {"arch": "native"}
        assert backend.normalize_options({"arch": "mlu590"}) == {"arch": "compute_50"}
        assert backend.normalize_options({"task_chunk": 512}) == {
            "arch": "native",
            "task_chunk": 512,
        }

        with pytest.raises(ValueError, match="BangC `arch`"):
            backend.normalize_options({"arch": "sm_90"})

        with pytest.raises(ValueError, match="must be positive"):
            backend.normalize_options({"task_chunk": 0})

    def test_bangc_arch_aliases_resolve(self):
        assert normalize_bangc_arch("MLU590") == "compute_50"
        assert normalize_bangc_arch("mtp_592") == "compute_50"
        assert normalize_bangc_arch("mlu370") == "compute_30"
        assert normalize_bangc_arch("native") == "native"

        with pytest.raises(ValueError, match="BangC `arch`"):
            normalize_bangc_arch("mlu100")

    def test_bangc_compile_command_uses_cncc(self):
        command = bangc_compile_command(
            "kernel.mlu",
            "kernel.so",
            arch="mlu590",
            cncc="/opt/neuware/bin/cncc",
        )
        assert command[0] == "/opt/neuware/bin/cncc"
        assert "--bang-arch=compute_50" in command
        assert "--shared" in command

    def test_bangc_rejects_source_only_kernel(self):
        with pytest.raises(ValueError, match="requires ssa.Program"):
            emit(_source_only_kernel(), "bangc")

    def test_bangc_elementwise_add_uses_vectorized_nram_staging(self):
        artifact = emit(_add_kernel(), "bangc")

        assert artifact.language == "bangc/c++"
        source = artifact.primary_source
        assert "__mlu_entry__ void add_kernel(" in source
        assert "__nram__ float nt_buf_x[16384];" in source
        assert "__nram__ float nt_buf_y[16384];" in source
        assert "__nram__ float nt_buf_out[16384];" in source
        assert "__memcpy(nt_buf_x, x + nt_base," in source
        assert "__bang_add(nt_buf_out, nt_buf_x, nt_buf_y, nt_aligned);" in source
        assert "__memcpy(out + nt_base, nt_buf_out," in source
        assert 'extern "C" int launch_add(' in source
        assert "cnrtQueue_t queue" in source
        assert "cnrtDim3_t dim;" in source
        assert "cnrtFuncTypeBlock" in source
        assert "<<<dim, ktype, queue>>>" in source
        assert "#include <bang.h>" in source

    def test_bangc_elementwise_chain_stages_without_bang_op(self):
        from ninetoothed.frontend.python import from_source
        from ninetoothed.ir import Kernel, TensorSpec

        tensors = (
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="y"),
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
        )
        program = from_source(
            "\ndef fused(x, y, out):\n    t = (x + y) * 2.0\n    out = t / 3.0\n",
            tensors,
            kind="fused",
        )
        kernel = Kernel(kernel_name="fused", source=None, tensors=tensors, ssa=program)
        source = emit(kernel, "bangc").primary_source

        assert "__nram__ float nt_buf_x[16384];" in source
        assert "__memcpy(nt_buf_x, x + nt_base," in source
        assert "__memcpy(out + nt_base, nt_buf_out," in source
        assert "for (int64_t nt_j = 0; nt_j < nt_cnt; nt_j++)" in source
        assert "__bang_add" not in source

    def test_bangc_half_elementwise_keeps_scalar_task_domain(self):
        artifact = emit(_add_kernel("float16"), "bangc")
        source = artifact.primary_source

        assert "__nram__" not in source
        assert "(int64_t)(taskIdX) * nt_chunk + nt_lane" in source
        assert "out[index] = v0;" in source

    def test_bangc_task_chunk_option_changes_domain(self):
        kernel = replace(
            _add_kernel(),
            compiler_options={"backend_options": {"task_chunk": 128}},
        )
        artifact = emit(kernel, "bangc")

        assert "const int64_t nt_chunk = 128;" in artifact.primary_source

    def test_bangc_staging_chunk_is_clamped_to_nram_budget(self):
        kernel = replace(
            _add_kernel(),
            compiler_options={"backend_options": {"task_chunk": 1 << 20}},
        )
        source = emit(kernel, "bangc").primary_source

        assert "__nram__ float nt_buf_x[16384];" in source
        assert "const int64_t nt_chunk = 16384;" in source

    def test_bangc_staging_default_uses_full_budget_chunk(self):
        source = emit(_add_kernel(), "bangc").primary_source

        assert "const int64_t nt_chunk = 16384;" in source

    def test_bangc_matmul_decomposes_into_scalar_loops(self):
        artifact = emit(_matmul_kernel(), "bangc")
        source = artifact.primary_source

        assert "for (int64_t v10_i = 0; v10_i < k; v10_i += 1)" in source
        assert "linalg.matmul" not in source
        assert "wmma::" not in source

    def test_bangc_fp8_is_rejected(self):
        with pytest.raises(ValueError, match="float8"):
            emit(_add_kernel("float8_e4m3fn"), "bangc")

    def test_bangc_rejects_vector_program_schedules(self):
        from ninetoothed.backends.emitters.bangc import BangCTarget
        from ninetoothed.backends.emitters.base import ModuleRenderContext

        context = object.__new__(ModuleRenderContext)
        object.__setattr__(context, "vector_program", True)
        object.__setattr__(context, "kernel", None)

        with pytest.raises(ValueError, match="flat-index task domain"):
            BangCTarget().render_module(context)

    def test_bangc_proposes_cooperative_reduction_candidate(self):
        from ninetoothed.backends.bangc import BangCOptimizeSchedule
        from ninetoothed.compiler.passes import Context

        candidates = BangCOptimizeSchedule().schedule_candidates(
            {},
            {
                "granularity": "parallel-reduction",
                "reduction": {"mode": "row-vector", "axis": 1, "extent": 64},
            },
            Context(
                backend=Target.BANGC,
                compiler_options={},
                kernel_metadata={},
            ),
        )
        assert tuple(candidate.name for candidate in candidates) == (
            "cooperative-reduction",
        )
        assert candidates[0].schedule == {"bangc_cooperative_reduction": True}

    def test_bangc_supports_cooperative_reduction_schedule(self):
        from ninetoothed.backends.emitters.bangc import TARGET

        assert TARGET.supports_cooperative_reduction(
            {"bangc_cooperative_reduction": True}
        )
        assert not TARGET.supports_cooperative_reduction({})
        assert TARGET.thread_id() == "0"
        assert TARGET.thread_count() == "1"

    def test_bangc_serializes_atomic_kernels_onto_one_task(self):
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
        artifact = emit(kernel, "bangc")
        source = artifact.primary_source

        assert "ninetoothed_atomic_add_f32(" in source
        assert "for (int64_t index = 0; index <" in source
        assert "int64_t nt_tasks = 1;" in source
        assert "taskIdX" not in source.split("__mlu_entry__")[1].split("extern")[0]

    def test_bangc_resolves_native_arch_with_capability_probe(self, monkeypatch):
        import ninetoothed.backends.toolchain as toolchain

        monkeypatch.delenv("NINETOOTHED_BANGC_ARCH", raising=False)

        class FakeMlu:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def get_device_capability(index):
                return (5, 0)

        class FakeTorch:
            mlu = FakeMlu

        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(**{"mlu": FakeMlu}))

        assert toolchain.resolve_bangc_arch("native") == "compute_50"

        class OldMlu:
            @staticmethod
            def is_available():
                return True

        class OldTorch:
            mlu = OldMlu

        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(mlu=OldMlu))
        assert toolchain.resolve_bangc_arch("native") == "compute_50"

        class NoMluTorch:
            pass

        monkeypatch.setitem(sys.modules, "torch", NoMluTorch)
        assert toolchain.resolve_bangc_arch("native") == "compute_50"
