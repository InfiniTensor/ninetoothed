import json
from dataclasses import replace

import pytest

import ninetoothed.backends.toolchain as toolchain
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
from ninetoothed.targets import PlatformProfile, TargetContext
from tests.utils import backend_platform_available, requires_backend

# Shared emission checks iterate every backend whose toolchain platform is
# reachable on this host.  Platform gating itself lives in
# tests/utils.py::_BACKEND_PLATFORM_PROBES; new backends only add a fragment
# entry below and (when platform-locked) a probe entry there.
EMISSION_TEST_BACKENDS = tuple(
    backend
    for backend in ("triton", "cuda", "tilelang", "bangc")
    if backend_platform_available(backend)
)


def _generic_target_context(backend):
    """Build a generic-platform context so emission snapshots stay hermetic.

    Without this, a pinned ``NINETOOTHED_PLATFORM`` in the environment would
    change platform constraints (grid limits, config caps) and break
    assertions that describe the generic lowering.
    """
    from ninetoothed.targets import default_platform_registry

    return TargetContext(
        backend=normalize_target(backend),
        platform=default_platform_registry().get("generic"),
    )


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


def _elementwise_kernel(tensor_count: int) -> Kernel:
    input_names = tuple(f"x{index}" for index in range(tensor_count - 1))
    parameters = ", ".join(input_names + ("out",))
    expression = " + ".join(input_names)
    tensors = tuple(
        TensorSpec(
            ndim=1,
            shape=("n",),
            dtype="float32",
            name=name,
            attrs={
                "source_name": name,
                "source_ndim": 1,
                "source_shape": ("n",),
                "source_strides": (f"{name}_stride_0",),
            },
        )
        for name in input_names + ("out",)
    )

    return _kernel_from_source(
        f"\ndef elementwise({parameters}):\n    out = {expression}\n",
        name="elementwise",
        tensors=tensors,
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

    @pytest.mark.parametrize("environment", ("CUDA_PATH", "CUCC_PATH"))
    def test_cuda_compiler_discovery_accepts_vendor_cucc(
        self, environment, monkeypatch, tmp_path
    ):
        compiler = tmp_path / "bin" / "cucc"
        compiler.parent.mkdir()
        compiler.write_text("vendor compiler", encoding="utf-8")
        compiler.chmod(0o755)
        monkeypatch.setattr(toolchain.shutil, "which", lambda name: None)
        monkeypatch.delenv("NINETOOTHED_CUDA_COMPILER", raising=False)
        monkeypatch.delenv("CUDA_HOME", raising=False)
        monkeypatch.delenv("CUDA_PATH", raising=False)
        monkeypatch.delenv("CUCC_PATH", raising=False)
        monkeypatch.setenv(environment, str(tmp_path))

        assert toolchain.find_nvcc() == str(compiler)

    def test_cuda_compiler_discovery_prefers_explicit_path(self, monkeypatch, tmp_path):
        compiler = tmp_path / "cucc"
        compiler.write_text("vendor compiler", encoding="utf-8")
        compiler.chmod(0o755)
        monkeypatch.setenv("NINETOOTHED_CUDA_COMPILER", str(compiler))
        monkeypatch.setattr(toolchain.shutil, "which", lambda name: None)

        assert toolchain.find_nvcc() == str(compiler)

    def test_invalid_explicit_cuda_compiler_does_not_fall_back(
        self, monkeypatch, tmp_path
    ):
        fallback = tmp_path / "nvcc"
        fallback.write_text("fallback compiler", encoding="utf-8")
        fallback.chmod(0o755)
        monkeypatch.setenv("NINETOOTHED_CUDA_COMPILER", str(tmp_path / "missing-cucc"))
        monkeypatch.setattr(
            toolchain.shutil,
            "which",
            lambda name: str(fallback) if name == "nvcc" else None,
        )

        with pytest.raises(RuntimeError, match="Explicit CUDA compiler"):
            toolchain.find_nvcc()

    def test_default_registry_reports_builtin_backends(self):
        names = {capability.name for capability in backend_capabilities()}
        assert names == {
            Target.TRITON,
            Target.TILELANG,
            Target.CUDA,
            Target.BANGC,
        }

    def test_backends_reject_source_only_kernel_without_ssa(self):
        for backend in EMISSION_TEST_BACKENDS:
            with pytest.raises(ValueError, match="requires ssa.Program"):
                emit(_source_only_kernel(), backend)

    def test_backends_emit_ssa_elementwise_add(self):
        expected = {
            "triton": ("python/triton", "tl.store(out + index, v0, mask=mask)"),
            "cuda": ("cuda/c++", "out[index] = v0;"),
            "tilelang": ("python/tilelang", "out_buf[index] = v0"),
            "bangc": (
                "bangc/c++",
                "__bang_add(nt_buf_out, nt_buf_x, nt_buf_y, nt_aligned);",
            ),
        }

        for backend in EMISSION_TEST_BACKENDS:
            language, fragment = expected[backend]
            artifact = emit(
                _add_kernel(), backend, target_context=_generic_target_context(backend)
            )
            assert artifact.language == language
            assert artifact.metadata["lowering_ir"] == "ssa.Program"
            assert artifact.metadata["target"]["platform"] == "generic"
            assert artifact.metadata["ssa_metadata"]["target_backend"] == backend
            assert fragment in artifact.primary_source
            assert "NotImplementedError" not in artifact.primary_source

    @pytest.mark.parametrize("tensor_count", (2, 3, 4))
    def test_triton_contiguous_predicates_are_left_associated(self, tensor_count):
        artifact = emit(_elementwise_kernel(tensor_count), "triton")
        names = tuple(f"x{index}" for index in range(tensor_count - 1)) + ("out",)
        predicates = tuple(f"({name}_stride_0 == 1)" for name in names)
        expected = predicates[0]

        for predicate in predicates[1:]:
            expected = f"({expected} and {predicate})"

        assert f"if {expected}:" in artifact.primary_source

    @requires_backend("cuda")
    def test_cuda_backend_includes_fp16_header_for_half_artifacts(self):
        artifact = emit(_add_kernel("float16"), "cuda")
        assert "#include <cuda_fp16.h>" in artifact.primary_source
        assert "const half* __restrict__ x" in artifact.primary_source
        assert "half* __restrict__ out" in artifact.primary_source

    @requires_backend("bangc")
    def test_bangc_backend_maps_half_dtypes_to_bang_types(self):
        artifact = emit(_add_kernel("float16"), "bangc")
        assert "#include <bang.h>" in artifact.primary_source
        assert "const half* __restrict__ x" in artifact.primary_source
        assert "half* __restrict__ out" in artifact.primary_source
        assert "__mlu_entry__" in artifact.primary_source
        assert "cnrtQueue_t queue" in artifact.primary_source

    def test_linalg_matmul_is_decomposed_before_backend_emission(self):
        expected_fragments = {
            "triton": "for v10_i in range(0, k, 1):",
            "cuda": "for (int64_t v10_i = 0; v10_i < k; v10_i += 1)",
            "tilelang": "for v10_i in T.serial(k)",
            "bangc": "for (int64_t v10_i = 0; v10_i < k; v10_i += 1)",
        }

        for backend in EMISSION_TEST_BACKENDS:
            artifact = emit(
                _matmul_kernel(),
                backend,
                target_context=_generic_target_context(backend),
            )
            assert expected_fragments[backend] in artifact.primary_source
            assert "linalg.matmul" not in artifact.primary_source

    @requires_backend("cuda")
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

    @requires_backend("cuda")
    def test_cuda_constraints_fall_back_to_generic_dot(self):
        kernel = replace(
            _matmul_kernel("float16"),
            compiler_options={"backend_options": {"arch": "sm_60"}},
        )
        artifact = emit(kernel, "cuda")
        assert "wmma::" not in artifact.primary_source
        rejected = artifact.metadata["ssa_metadata"]["rejected_schedule_candidates"]
        assert "requires compute capability 7.0" in rejected[0]["reason"]

    @requires_backend("cuda")
    def test_cuda_profile_can_disable_unverified_wmma(self):
        target = TargetContext(
            backend=Target.CUDA,
            platform=PlatformProfile(
                name="vendor-cuda",
                compute_arch="vendor-native",
                backend_modes={"cuda": frozenset({"jit"})},
                metadata={"cuda": {"arch": "native", "wmma": False}},
            ),
        )
        artifact = emit(
            _matmul_kernel("float16"),
            "cuda",
            target_context=target,
        )

        assert "wmma::" not in artifact.primary_source
        assert not artifact.metadata["ssa_metadata"]["schedule_candidates"]

    @requires_backend("cuda")
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

    @requires_backend("cuda")
    def test_generic_cuda_launch_matches_emitted_thread_count(self):
        artifact = emit(_add_kernel(), "cuda")
        assert "constexpr int threads = 256;" in artifact.primary_source
        assert "if (blocks <= 0)" in artifact.primary_source
        assert artifact.metadata["launch_block"] == ("256",)

    @requires_backend("cuda")
    def test_artifact_can_write_all_sources(self, tmp_path):
        artifact = emit(_add_kernel(), "cuda")
        paths = artifact.write_to(tmp_path)
        assert len(paths) == 2
        assert all((path.exists() for path in paths))

    def test_replaced_profile_revalidates_existing_targeted_ssa(self):
        kernel = _add_kernel()
        program = replace(
            kernel.ssa,
            metadata=dict(kernel.ssa.metadata)
            | {"required_capabilities": ("dtype.fp8",)},
        )
        supported = TargetContext(
            backend=Target.TRITON,
            platform=PlatformProfile(
                name="replaceable",
                backend_modes={"triton": frozenset({"jit", "aot"})},
                supported_capabilities=frozenset({"dtype.fp8"}),
            ),
        )
        first = emit(replace(kernel, ssa=program), target_context=supported)
        unsupported = TargetContext(
            backend=Target.TRITON,
            platform=PlatformProfile(
                name="replaceable",
                backend_modes={"triton": frozenset({"jit", "aot"})},
                unsupported_capabilities=frozenset({"dtype.fp8"}),
            ),
        )

        with pytest.raises(ValueError, match="does not support `dtype.fp8`"):
            emit(
                replace(
                    kernel,
                    ssa=replace(
                        kernel.ssa,
                        metadata=dict(first.metadata["ssa_metadata"]),
                    ),
                ),
                target_context=unsupported,
            )

    def test_same_target_revalidates_mutated_ssa_capabilities(self):
        kernel = _add_kernel()
        target = TargetContext(
            backend=Target.TRITON,
            platform=PlatformProfile(
                name="unsupported-profile",
                backend_modes={"triton": frozenset({"jit", "aot"})},
                unsupported_capabilities=frozenset({"dtype.fp8"}),
            ),
        )
        first = emit(kernel, target_context=target)
        reused = replace(
            kernel.ssa,
            metadata=dict(first.metadata["ssa_metadata"])
            | {"required_capabilities": ("dtype.fp8",)},
        )

        with pytest.raises(ValueError, match="does not support `dtype.fp8`"):
            emit(replace(kernel, ssa=reused), target_context=target)

    @requires_backend("cuda")
    def test_manifest_and_in_memory_metadata_share_one_builder(self):
        artifact = emit(_add_kernel(), "cuda")
        manifest = next(
            content
            for name, content in artifact.sources.items()
            if name.endswith(".json")
        )

        assert json.loads(manifest) == ir_to_dict(artifact.metadata)
