import json
from dataclasses import replace

import pytest

from ninetoothed.backends import (
    Target,
    backend_capabilities,
    default_registry,
    emit,
    normalize_target,
)
from ninetoothed.backends.toolchain import cuda_compile_command
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import Kernel, TensorSpec, ir_to_dict


def _source_only_kernel():
    return Kernel(
        kernel_name="add",
        source="@triton.jit\ndef add(x, y, out):\n    return\n",
        entrypoint="add",
        tensors=(
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="y"),
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
        ),
        compiler_options={"num_warps": 4, "num_stages": 3},
    )


def _kernel_from_source(
    source: str,
    *,
    name: str,
    tensors: tuple[TensorSpec, ...],
) -> Kernel:
    program = from_source(source, tensors, kind=name)
    assert program is not None

    return Kernel(
        kernel_name=name,
        source=source,
        source_language="ninetoothed-python",
        entrypoint=name,
        tensors=tensors,
        ssa=program,
    )


def _add_kernel(dtype: str = "float32") -> Kernel:
    return _kernel_from_source(
        "\ndef add(x, y, out):\n    out = x + y\n",
        name="add",
        tensors=(
            TensorSpec(ndim=1, shape=("n",), dtype=dtype, name="x"),
            TensorSpec(ndim=1, shape=("n",), dtype=dtype, name="y"),
            TensorSpec(ndim=1, shape=("n",), dtype=dtype, name="out"),
        ),
    )


def _matmul_kernel(dtype: str = "float32") -> Kernel:
    return _kernel_from_source(
        "\ndef matmul(a, b, out):\n    out = a @ b\n",
        name="matmul",
        tensors=(
            TensorSpec(ndim=2, shape=("m", "k"), dtype=dtype, name="a"),
            TensorSpec(ndim=2, shape=("k", "n"), dtype=dtype, name="b"),
            TensorSpec(ndim=2, shape=("m", "n"), dtype=dtype, name="out"),
        ),
    )


class TestRegistry:
    def test_backend_names_are_normalized_without_aliases(self):
        assert normalize_target(None) == Target.TRITON
        assert normalize_target("triton") == Target.TRITON
        assert normalize_target("tilelang") == Target.TILELANG
        assert normalize_target("cuda") == Target.CUDA

        for alias in ("tl", "tile-lang", "tile_lang", "cu"):
            with pytest.raises(ValueError, match="Unsupported backend"):
                normalize_target(alias)

        with pytest.raises(ValueError, match="Unsupported backend"):
            normalize_target("tvm")

    def test_backend_options_are_validated_and_normalized_by_target(self):
        cuda = default_registry().get(Target.CUDA)
        options = cuda.normalize_options({"arch": "SM_90"})
        assert options == {"arch": "sm_90", "compute_capability": "9.0"}

        with pytest.raises(TypeError, match="Unsupported cuda backend option"):
            cuda.normalize_options({"caller": "cuda"})

        with pytest.raises(TypeError, match="Unsupported triton backend option"):
            emit(
                replace(
                    _add_kernel(),
                    compiler_options={"backend_options": {"arch": "sm_90"}},
                ),
                "triton",
            )

    def test_cuda_arch_is_materialized_in_nvcc_command(self):
        command = cuda_compile_command(
            "kernel.cu",
            "kernel.so",
            arch="sm_90",
            nvcc="/opt/cuda/bin/nvcc",
        )
        assert "-arch=sm_90" in command

    def test_default_registry_reports_three_backends(self):
        names = {capability.name for capability in backend_capabilities()}
        assert names == {
            Target.TRITON,
            Target.TILELANG,
            Target.CUDA,
        }

    def test_backends_reject_source_only_kernel_without_ssa(self):
        for backend in ("triton", "cuda", "tilelang"):
            with pytest.raises(ValueError, match="requires ssa.Program"):
                emit(_source_only_kernel(), backend)

    def test_backends_emit_ssa_elementwise_add(self):
        expected = {
            "triton": ("python/triton", "tl.store(out + index, v0, mask=mask)"),
            "cuda": ("cuda/c++", "out[index] = v0;"),
            "tilelang": ("python/tilelang", "out_buf[index] = v0"),
        }

        for backend, (language, fragment) in expected.items():
            artifact = emit(_add_kernel(), backend)
            assert artifact.language == language
            assert artifact.metadata["lowering_ir"] == "ssa.Program"
            assert artifact.metadata["ssa_metadata"]["target_backend"] == backend
            assert fragment in artifact.primary_source
            assert "NotImplementedError" not in artifact.primary_source

    def test_cuda_backend_includes_fp16_header_for_half_artifacts(self):
        artifact = emit(_add_kernel("float16"), "cuda")
        assert "#include <cuda_fp16.h>" in artifact.primary_source
        assert "const half* __restrict__ x" in artifact.primary_source
        assert "half* __restrict__ out" in artifact.primary_source

    def test_linalg_matmul_is_decomposed_before_backend_emission(self):
        expected_fragments = {
            "triton": "for v10_i in range(0, k, 1):",
            "cuda": "for (int64_t v10_i = 0; v10_i < k; v10_i += 1)",
            "tilelang": "for v10_i in T.serial(k)",
        }

        for backend, fragment in expected_fragments.items():
            artifact = emit(_matmul_kernel(), backend)
            assert fragment in artifact.primary_source
            assert "linalg.matmul" not in artifact.primary_source

    def test_cuda_exposes_only_materialized_wmma_schedule(self):
        artifact = emit(_matmul_kernel("float16"), "cuda")
        candidates = artifact.metadata["ssa_metadata"]["schedule_candidates"]
        assert tuple(candidate["name"] for candidate in candidates) == ("wmma-16x16",)

        kernel = replace(
            _matmul_kernel("float16"),
            compiler_options={
                "ssa_pass_options": {
                    "ssa.cuda.optimize_schedule": {"candidate": "wmma-32x32"}
                }
            },
        )

        with pytest.raises(ValueError, match="Unknown schedule candidate"):
            emit(kernel, "cuda")

    def test_cuda_constraints_fall_back_to_generic_dot(self):
        kernel = replace(
            _matmul_kernel("float16"),
            compiler_options={"backend_options": {"arch": "sm_60"}},
        )
        artifact = emit(kernel, "cuda")
        assert "wmma::" not in artifact.primary_source
        rejected = artifact.metadata["ssa_metadata"]["rejected_schedule_candidates"]
        assert "requires compute capability 7.0" in rejected[0]["reason"]

    def test_optimization_metadata_contains_only_materialized_choices(self):
        forbidden_fields = {
            "input_precision",
            "lowering",
            "passes",
            "use_tensor_cores",
            "vector_width",
        }

        for backend in Target:
            artifact = emit(_matmul_kernel("float16"), backend)
            optimization = artifact.metadata["ssa_optimization"]
            assert set(optimization) <= {"preserve_linalg", "schedule"}
            assert forbidden_fields.isdisjoint(optimization)
            assert "small_problem_" not in json.dumps(
                ir_to_dict(artifact.metadata["ssa"])
            )

    def test_generic_cuda_launch_matches_emitted_thread_count(self):
        artifact = emit(_add_kernel(), "cuda")
        assert "constexpr int threads = 256;" in artifact.primary_source
        assert "if (blocks <= 0)" in artifact.primary_source
        assert artifact.metadata["launch_block"] == ("256",)

    def test_artifact_can_write_all_sources(self, tmp_path):
        artifact = emit(_add_kernel(), "cuda")
        paths = artifact.write_to(tmp_path)
        assert len(paths) == 2
        assert all((path.exists() for path in paths))

    def test_manifest_and_in_memory_metadata_share_one_builder(self):
        artifact = emit(_add_kernel(), "cuda")
        manifest = next(
            content
            for name, content in artifact.sources.items()
            if name.endswith(".json")
        )

        assert json.loads(manifest) == ir_to_dict(artifact.metadata)
