import importlib
import os
import subprocess
import sys
from dataclasses import replace

import pytest

import ninetoothed
from ninetoothed import Tensor
from ninetoothed.compiler import DEFAULT_COMPILER, Compiler, CompileRequest
from ninetoothed.compiler import driver as compiler_driver
from ninetoothed.targets import resolve_target_context


def _arrangement(input, other, output):
    return tuple(tensor.tile((64,)) for tensor in (input, other, output))


def _application(input, other, output):
    output = input + other  # noqa: F841


def test_public_entrypoints_remain_conservative():
    assert isinstance(DEFAULT_COMPILER, Compiler)
    assert callable(ninetoothed.build)
    assert callable(ninetoothed.jit)
    assert callable(ninetoothed.make)
    assert not hasattr(ninetoothed, "load_built_artifact")


def test_legacy_entrypoint_modules_remain_importable():
    assert callable(importlib.import_module("ninetoothed.aot").aot)
    assert callable(importlib.import_module("ninetoothed.make").make)


def test_jit_implementation_class_is_not_public():
    compiler = importlib.import_module("ninetoothed.compiler")

    assert not hasattr(compiler, "JIT")
    assert "JIT" not in compiler.__all__


def test_jit_function_and_decorator_use_the_default_compiler(monkeypatch):
    requests = []

    def materialize(request, *, output_dir=None, mode="jit"):
        requests.append((request, output_dir, mode))

        return request.backend

    monkeypatch.setattr(DEFAULT_COMPILER, "materialize", materialize)

    assert ninetoothed.jit(_application, backend="cuda", arch="sm_90") == "cuda"
    assert ninetoothed.jit(backend="tilelang")(_application) == "tilelang"
    assert (
        ninetoothed.jit(
            _application,
            backend="cuda",
            platform="generic",
            compute_arch="sm_90",
            arch="sm_90",
        )
        == "cuda"
    )
    assert requests[0][0].kernel_name == "_application"
    assert requests[0][0].platform is None
    assert requests[0][0].compute_arch is None
    assert requests[0][0].backend_options == {"arch": "sm_90"}
    assert requests[1][2] == "jit"
    assert requests[2][0].platform == "generic"
    assert requests[2][0].compute_arch == "sm_90"


def test_pure_lowering_does_not_query_runtime_cuda_architecture(monkeypatch):
    import ninetoothed.compiler.cache as cache

    def unexpected_query():
        raise AssertionError("Pure lowering must not query a runtime device.")

    monkeypatch.setattr(cache, "_runtime_cuda_architecture", unexpected_query)
    artifact = compiler_driver.lower(
        _arrangement,
        _application,
        (Tensor(1), Tensor(1), Tensor(1)),
        backend="cuda",
        platform="generic",
        compute_arch="sm_90",
    )

    assert artifact.backend.value == "cuda"
    assert artifact.metadata["target"]["platform"] == "generic"
    assert artifact.metadata["target"]["compute_arch"] == "sm_90"


def test_make_preserves_legacy_caller_materialization_mode(tmp_path, monkeypatch):
    calls = []

    def materialize(request, *, output_dir=None, mode="jit"):
        calls.append((request, output_dir, mode))

        return mode

    monkeypatch.setattr(DEFAULT_COMPILER, "materialize", materialize)
    tensors = (Tensor(1), Tensor(1), Tensor(1))

    assert ninetoothed.make(_arrangement, _application, tensors) == "jit"
    assert (
        ninetoothed.make(
            _arrangement,
            _application,
            tensors,
            "cuda",
            "legacy_aot",
            tmp_path,
        )
        == "aot"
    )
    assert (
        ninetoothed.make(_arrangement, _application, tensors, platform="generic")
        == "jit"
    )
    assert calls[0][2] == "jit"
    assert calls[0][0].platform is None
    assert calls[1][0].caller == "cuda"
    assert calls[1][1] == tmp_path
    assert calls[1][2] == "aot"
    assert calls[2][0].platform == "generic"


def test_compilation_rejects_target_and_artifact_backend_mismatch():
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=(Tensor(1), Tensor(1), Tensor(1)),
            backend="triton",
        )
    )

    with pytest.raises(ValueError, match="does not match artifact backend"):
        replace(compilation, target=resolve_target_context("cuda"))


def test_explicit_cuda_arch_binds_and_validates_backend_architecture():
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=(Tensor(1), Tensor(1), Tensor(1)),
            backend="cuda",
            platform="generic",
            compute_arch="sm_90",
        )
    )

    assert compilation.request.backend_options["arch"] == "sm_90"
    assert compilation.request.backend_options["compute_capability"] == "9.0"
    assert compilation.kernel.compiler_options["backend_options"]["arch"] == "sm_90"

    with pytest.raises(ValueError, match="conflicts with target architecture"):
        DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=_arrangement,
                application=_application,
                tensors=(Tensor(1), Tensor(1), Tensor(1)),
                backend="cuda",
                platform="generic",
                compute_arch="sm_90",
                backend_options={"arch": "sm_80"},
            )
        )


def test_compilation_rejects_target_replacement_with_same_backend():
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=(Tensor(1), Tensor(1), Tensor(1)),
            backend="triton",
            platform="generic",
            compute_arch="arch-a",
        )
    )

    with pytest.raises(ValueError, match="target metadata does not match"):
        replace(
            compilation,
            target=resolve_target_context(
                "triton",
                platform="generic",
                compute_arch="arch-b",
            ),
        )


def test_triton_launch_plan_enumerates_symbolic_meta_parameters():
    block = ninetoothed.block_size(lower_bound=32, upper_bound=64)

    def arrangement(input, other, output):
        return tuple(tensor.tile((block,)) for tensor in (input, other, output))

    compilation = compiler_driver.compile_kernel(
        CompileRequest(
            arrangement=arrangement,
            application=_application,
            tensors=(Tensor(1), Tensor(1), Tensor(1)),
            backend="triton",
        )
    )
    candidates = compilation.launch_plan.tuning_candidates

    assert len(candidates) == 2
    assert {
        next(iter(candidate["meta_parameters"].values())) for candidate in candidates
    } == {32, 64}


def test_package_import_does_not_create_compiler_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    env = dict(os.environ, NINETOOTHED_CACHE_DIR=str(cache_dir))
    subprocess.run(
        [sys.executable, "-c", "import ninetoothed"],
        check=True,
        env=env,
    )
    assert not cache_dir.exists()


def test_triton_launch_plan_contains_limited_runtime_variants():
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=(Tensor(1), Tensor(1), Tensor(1)),
            backend="triton",
            num_warps=(4, 8),
            num_stages=(2, 3),
            max_num_configs=3,
        )
    )

    assert compilation.launch_plan.tuning_candidates == (
        {"id": "warps-4_stages-2", "num_warps": 4, "num_stages": 2},
        {"id": "warps-4_stages-3", "num_warps": 4, "num_stages": 3},
        {"id": "warps-8_stages-2", "num_warps": 8, "num_stages": 2},
    )


@pytest.mark.parametrize("backend", ("cuda", "tilelang"))
@pytest.mark.parametrize(
    "options",
    (
        {"num_warps": (4, 8)},
        {"num_stages": (2, 3)},
        {"max_num_configs": 2},
    ),
)
def test_non_triton_backends_reject_unsupported_auto_tuning(backend, options):
    with pytest.raises(NotImplementedError, match="auto-tuning is not supported"):
        DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=_arrangement,
                application=_application,
                tensors=(Tensor(1), Tensor(1), Tensor(1)),
                backend=backend,
                **options,
            )
        )
