import pytest

from ninetoothed.backends import (
    Target,
    backend_capabilities,
    emit,
    normalize_options,
    normalize_target,
)
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import Kernel, Launch, TensorSpec


def _source_only_kernel():
    return Kernel(
        kernel_name="add",
        source="@triton.jit\ndef add(x, y, out):\n    return\n",
        entrypoint="add",
        launch=Launch(
            name="launch_add", args=("x", "y", "out"), grid="lambda meta: (1,)"
        ),
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


def _matmul_kernel() -> Kernel:
    return _kernel_from_source(
        "\ndef matmul(a, b, out):\n    out = a @ b\n",
        name="matmul",
        tensors=(
            TensorSpec(ndim=2, shape=("m", "k"), dtype="float32", name="a"),
            TensorSpec(ndim=2, shape=("k", "n"), dtype="float32", name="b"),
            TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="out"),
        ),
    )


class TestRegistry:
    def test_backend_names_are_normalized_without_aliases(self):
        assert normalize_target(None) == Target.TRITON
        assert normalize_target("triton") == Target.TRITON
        assert normalize_target("tilelang") == Target.TILELANG
        assert normalize_target("cuda") == Target.CUDA
        assert normalize_target("tvm") == Target.TVM

        for alias in ("tl", "tile-lang", "tile_lang", "cu", "tvm-script", "tvmscript"):
            with pytest.raises(ValueError, match="Unsupported backend"):
                normalize_target(alias)

    def test_backend_options_keep_caller_and_extra_values(self):
        options = normalize_options(
            "cuda", caller="cuda", emit_only=False, arch="sm_90"
        )
        assert options.name == Target.CUDA
        assert options.caller == "cuda"
        assert not options.emit_only
        assert options.extra["arch"] == "sm_90"

    def test_default_registry_reports_four_backends(self):
        names = {capability.name for capability in backend_capabilities()}
        assert names == {
            Target.TRITON,
            Target.TILELANG,
            Target.CUDA,
            Target.TVM,
        }

    def test_backends_reject_source_only_kernel_without_ssa(self):
        for backend in ("triton", "cuda", "tilelang", "tvm"):
            with pytest.raises(ValueError, match="requires ssa.Program"):
                emit(_source_only_kernel(), backend)

    def test_backends_emit_ssa_elementwise_add(self):
        expected = {
            "triton": ("python/triton", "tl.store(out + index, v0, mask=mask)"),
            "cuda": ("cuda/c++", "out[index] = v0;"),
            "tilelang": ("python/tilelang", "out_buf[index] = v0"),
            "tvm": ("python/tvm-script", "out_buf[index] = v0"),
        }

        for backend, (language, fragment) in expected.items():
            artifact = emit(_add_kernel(), backend)
            assert artifact.executable
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
            "tvm": "for v10_i in T.serial(k)",
        }

        for backend, fragment in expected_fragments.items():
            artifact = emit(_matmul_kernel(), backend)
            assert artifact.executable
            assert fragment in artifact.primary_source
            assert "linalg.matmul" not in artifact.primary_source

    def test_artifact_can_write_all_sources(self, tmp_path):
        artifact = emit(_add_kernel(), "cuda")
        paths = artifact.write_to(tmp_path)
        assert len(paths) == 2
        assert all((path.exists() for path in paths))
