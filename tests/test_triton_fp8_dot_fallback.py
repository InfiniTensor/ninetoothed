import ast
import importlib.metadata
import re
from types import SimpleNamespace

import pytest

import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor
from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters.triton import TritonTarget
from ninetoothed.backends.triton import TritonBackend
from ninetoothed.compiler import CompileRequest, compile_kernel
from ninetoothed.compiler.cache import compilation_cache_key
from ninetoothed.ir import ssa

BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)
BLOCK_SIZE_K = Symbol("BLOCK_SIZE_K", meta=True)


def _matmul_arrangement(
    lhs,
    rhs,
    output,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    output_tiled = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))
    lhs_tiled = (
        lhs.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
        .tile((1, -1))
        .expand((-1, output_tiled.shape[1]))
    )
    lhs_tiled.dtype = lhs_tiled.dtype.squeeze(0)
    rhs_tiled = (
        rhs.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
        .tile((-1, 1))
        .expand((output_tiled.shape[0], -1))
    )
    rhs_tiled.dtype = rhs_tiled.dtype.squeeze(1)

    return lhs_tiled, rhs_tiled, output_tiled


def _matmul_application(lhs, rhs, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(lhs.shape[0]):
        accumulator += ntl.dot(lhs[k], rhs[k])

    output = accumulator.to(ntl.float16)  # noqa: F841


def _matmul_tensors(dtype):
    return (
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=dtype),
        Tensor(2, dtype="float16"),
    )


def _compile_matmul(dtype, backend_options=None, *, tensors=None):
    return compile_kernel(
        CompileRequest(
            arrangement=_matmul_arrangement,
            application=_matmul_application,
            tensors=tensors or _matmul_tensors(dtype),
            backend="triton",
            backend_options=backend_options,
        )
    )


def _dot_operation(dtype):
    operand_type = ssa.Type(kind="tensor", shape=("M", "K"), dtype=dtype)

    return ssa.Operation(
        opcode="linalg.dot",
        operands=("%lhs", "%rhs"),
        results=(
            ssa.Value(
                name="%result",
                type=ssa.Type(kind="tensor", shape=("M", "N"), dtype="float32"),
            ),
        ),
    ), {
        "%lhs": operand_type,
        "%rhs": operand_type,
    }


def _coercion_context(value_types, target, *, local_suffix=""):
    return SimpleNamespace(
        target=target,
        value_types=value_types,
        lines=[],
        temp_counter=[0],
        local_suffix=local_suffix,
    )


def _cache_compilation(backend_options):
    return SimpleNamespace(
        request=SimpleNamespace(
            backend_options=backend_options,
            pipeline=None,
            pass_options=None,
        ),
        artifact=SimpleNamespace(
            backend=Target.TRITON,
            sources={"kernel": "source"},
        ),
        kernel=SimpleNamespace(
            ssa=(),
            compiler_options={"backend_options": backend_options},
        ),
        launch_plan=(),
    )


@pytest.mark.parametrize(
    "triton_version, expected",
    (
        ("3.1.0+corex.4.4.0", "float16"),
        ("3.5.1", "none"),
        ("3.3.0+rocm6.3", "none"),
    ),
)
@pytest.mark.parametrize("options", ({}, {"fp8_dot_fallback": "auto"}))
def test_auto_fallback_uses_triton_distribution_metadata(
    triton_version, expected, options, monkeypatch
):
    calls = []

    def distribution_version(package):
        calls.append(package)

        return triton_version

    monkeypatch.setattr(importlib.metadata, "version", distribution_version)

    assert TritonBackend().normalize_options(options) == {"fp8_dot_fallback": expected}
    assert calls == ["triton"]


def test_auto_fallback_defaults_to_none_without_distribution_metadata(monkeypatch):
    def missing_distribution(package):
        raise importlib.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(importlib.metadata, "version", missing_distribution)

    assert TritonBackend().normalize_options({}) == {"fp8_dot_fallback": "none"}


@pytest.mark.parametrize("fallback", ("none", "float16"))
def test_explicit_fallback_does_not_query_triton_metadata(fallback, monkeypatch):
    def unexpected_query(package):
        raise AssertionError(f"Explicit fallback queried `{package}` metadata.")

    monkeypatch.setattr(importlib.metadata, "version", unexpected_query)

    assert TritonBackend().normalize_options({"fp8_dot_fallback": fallback}) == {
        "fp8_dot_fallback": fallback
    }


def test_invalid_fallback_is_rejected():
    backend = TritonBackend()

    with pytest.raises(TypeError, match="must be a string"):
        backend.normalize_options({"fp8_dot_fallback": None})

    with pytest.raises(ValueError, match="`auto`, `none`, or `float16`"):
        backend.normalize_options({"fp8_dot_fallback": "fp16"})


def test_corex_auto_fallback_is_carried_into_compilation_and_guards_both_operands(
    monkeypatch,
):
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda package: "3.1.0+corex.4.4.0",
    )

    compilation = _compile_matmul("float8_e5m2")
    expected_options = {"fp8_dot_fallback": "float16"}
    assert compilation.request.backend_options == expected_options
    assert compilation.kernel.compiler_options["backend_options"] == expected_options

    source = compilation.artifact.primary_source
    guards = re.findall(
        r"if ([A-Za-z_]\w*_fp8_(?:lhs|rhs))\.dtype == tl\.float8e5:\n"
        r"\s+\1 = \1\.to\(tl\.float16\)",
        source,
    )
    dot = re.search(
        r"tl\.dot\(([A-Za-z_]\w*_fp8_lhs), ([A-Za-z_]\w*_fp8_rhs)\)",
        source,
    )
    assert len(guards) == 2
    assert dot is not None
    assert set(dot.groups()) == set(guards)
    ast.parse(source)


def test_explicit_float16_fallback_reaches_source_without_distribution_metadata(
    monkeypatch,
):
    def missing_distribution(package):
        raise importlib.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(importlib.metadata, "version", missing_distribution)

    compilation = _compile_matmul("float8_e5m2", {"fp8_dot_fallback": "float16"})

    assert compilation.request.backend_options == {"fp8_dot_fallback": "float16"}
    assert compilation.artifact.primary_source.count(".dtype == tl.float8e5") == 2


@pytest.mark.parametrize("triton_version", ("3.5.1", "3.3.0+rocm6.3", None))
def test_non_corex_auto_fallback_preserves_original_dot_source(
    triton_version, monkeypatch
):
    def distribution_version(package):
        if triton_version is None:
            raise importlib.metadata.PackageNotFoundError(package)

        return triton_version

    monkeypatch.setattr(importlib.metadata, "version", distribution_version)

    tensors = _matmul_tensors("float8_e5m2")
    compilation = _compile_matmul("float8_e5m2", tensors=tensors)
    explicit_none = _compile_matmul(
        "float8_e5m2",
        {"fp8_dot_fallback": "none"},
        tensors=tensors,
    )
    expected_options = {"fp8_dot_fallback": "none"}
    assert compilation.request.backend_options == expected_options
    assert compilation.kernel.compiler_options["backend_options"] == expected_options
    assert "tl.dot(" in compilation.artifact.primary_source
    assert ".dtype == tl.float8e5" not in compilation.artifact.primary_source
    assert compilation.artifact.primary_source == explicit_none.artifact.primary_source


def test_unknown_ssa_dtype_is_guarded_by_float16_fallback():
    operation, value_types = _dot_operation(None)
    target = TritonTarget(fp8_dot_fallback="float16")
    context = _coercion_context(value_types, target, local_suffix="_loop_body")
    operands = target.coerce_block_dot_operands(
        operation, ("lhs_value", "rhs_value"), context
    )
    source = "\n".join(
        (*context.lines, f"result = tl.dot({operands[0]}, {operands[1]})")
    )

    assert source.count(".dtype == tl.float8e5") == 2
    assert operands == (
        "vresult_loop_body_fp8_lhs",
        "vresult_loop_body_fp8_rhs",
    )
    ast.parse(source)


def test_known_float16_operands_keep_jit_static_guards():
    operation, value_types = _dot_operation("float16")
    target = TritonTarget(fp8_dot_fallback="float16")
    context = _coercion_context(value_types, target)
    operands = target.coerce_block_dot_operands(
        operation, ("lhs_value", "rhs_value"), context
    )

    assert operands == ("vresult_fp8_lhs", "vresult_fp8_rhs")
    assert "\n".join(context.lines).count(".dtype == tl.float8e5") == 2


def test_default_target_does_not_guard_block_dot_operands():
    operation, value_types = _dot_operation("float8_e5m2")
    target = TritonTarget()
    context = _coercion_context(value_types, target)
    operands = target.coerce_block_dot_operands(
        operation, ("lhs_value", "rhs_value"), context
    )

    assert operands == ("lhs_value", "rhs_value")
    assert not context.lines


def test_effective_fallback_is_part_of_compilation_cache_key(monkeypatch):
    import ninetoothed.compiler.cache as cache

    monkeypatch.setattr(cache, "_architecture", lambda compilation: {})
    monkeypatch.setattr(cache, "_compiler_versions", lambda: {"triton": "fixed"})
    backend = TritonBackend()
    none = backend.normalize_options({"fp8_dot_fallback": "none"})
    float16 = backend.normalize_options({"fp8_dot_fallback": "float16"})

    assert compilation_cache_key(_cache_compilation(none)) != compilation_cache_key(
        _cache_compilation(float16)
    )
