import pytest

import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor
from ninetoothed.backends.core import Target
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.targets import (
    PlatformProfile,
    PlatformRegistry,
    TargetContext,
    default_platform_registry,
    resolve_target_context,
    runtime_device_types,
)


def _registry(*profiles):
    registry = PlatformRegistry()

    for profile in profiles:
        registry.register(profile)

    return registry


def test_default_profiles_expose_normalized_independent_contracts():
    registry = default_platform_registry()
    profiles = registry.profiles()

    assert profiles
    assert len({profile.name for profile in profiles}) == len(profiles)

    for profile in profiles:
        assert profile.name == profile.name.strip().lower().replace("_", "-")
        assert "family" not in profile.as_metadata()
        assert registry.get(profile.name) is profile

        for alias in profile.aliases:
            assert registry.get(alias) is profile

        for backend, modes in profile.backend_modes.items():
            target = TargetContext(backend=backend, platform=profile)

            for mode in modes:
                target.validate_materialization(mode)


def test_target_context_normalizes_aliases_and_optional_compute_architecture():
    profile = PlatformProfile(
        name="accelerator-v1",
        aliases=("current",),
        compute_arch="arch-v1",
        device_types=("vendor",),
        backend_modes={"triton": frozenset({"jit"})},
    )
    registry = _registry(profile)
    target = resolve_target_context("triton", platform="CURRENT", registry=registry)

    assert target.backend == Target.TRITON
    assert target.platform is profile
    assert target.compute_arch == "arch-v1"
    assert target.device_types == ("vendor",)

    with pytest.raises(ValueError, match="conflicts with platform"):
        resolve_target_context(
            "triton",
            platform="accelerator-v1",
            compute_arch="arch-v2",
            registry=registry,
        )

    with pytest.raises(ValueError, match="conflicts with platform"):
        TargetContext(
            backend=Target.TRITON,
            platform=profile,
            compute_arch="arch-v2",
        )


def test_profiles_reject_unsupported_backend_pairings():
    triton_profile = PlatformProfile(
        name="jit-only",
        device_types=("vendor",),
        backend_modes={"triton": frozenset({"jit"})},
        backend_unsupported_capabilities={"triton": frozenset({"reduction.min"})},
    )
    cuda_profile = PlatformProfile(
        name="cuda-only",
        compute_arch="vendor-arch",
        device_types=("vendor",),
        backend_modes={"cuda": frozenset({"jit"})},
    )
    disabled_profile = PlatformProfile(name="disabled", device_types=("vendor",))
    registry = _registry(triton_profile, cuda_profile, disabled_profile)

    with pytest.raises(ValueError, match="Backend `cuda` is not supported"):
        resolve_target_context("cuda", platform="jit-only", registry=registry)

    with pytest.raises(ValueError, match="Backend `tilelang` is not supported"):
        resolve_target_context("tilelang", platform="jit-only", registry=registry)

    with pytest.raises(ValueError, match="Backend `triton` is not supported"):
        resolve_target_context("triton", platform="disabled", registry=registry)

    triton = resolve_target_context("triton", platform="jit-only", registry=registry)
    cuda = resolve_target_context("cuda", platform="cuda-only", registry=registry)

    assert runtime_device_types(type("Compilation", (), {"target": triton})()) == (
        "vendor",
    )
    assert runtime_device_types(type("Compilation", (), {"target": cuda})()) == (
        "cuda",
    )
    triton.validate_materialization("jit")

    with pytest.raises(ValueError, match="does not support `aot` materialization"):
        triton.validate_materialization("aot")

    with pytest.raises(ValueError, match="does not support .*reduction.min"):
        triton.validate_capabilities(("reduction.min",))


def test_capabilities_can_be_platform_or_backend_specific():
    profile = PlatformProfile(
        name="capability-scopes",
        backend_modes={
            "triton": frozenset({"jit"}),
            "cuda": frozenset({"jit"}),
        },
        unsupported_capabilities=frozenset({"math.pow"}),
        backend_unsupported_capabilities={"cuda": frozenset({"dtype.fp8"})},
    )
    triton = TargetContext(backend=Target.TRITON, platform=profile)
    cuda = TargetContext(backend=Target.CUDA, platform=profile)
    report = triton.validate_capabilities(("dtype.fp8",))

    assert report["supported"] == ()
    assert report["unsupported"] == ()
    assert report["unresolved"] == ("dtype.fp8",)

    with pytest.raises(ValueError, match="does not support `dtype.fp8`"):
        cuda.validate_capabilities(("dtype.fp8",))

    for target in (triton, cuda):
        with pytest.raises(ValueError, match="does not support `math.pow`"):
            target.validate_capabilities(("math.pow",))


def test_custom_profiles_require_explicit_backend_and_mode_opt_in():
    profile = PlatformProfile(name="vendor", device_types=("vendor",))

    for backend in Target:
        with pytest.raises(ValueError, match="is not supported by platform"):
            TargetContext(backend=backend, platform=profile)

    target = TargetContext(
        backend=Target.TRITON,
        platform=PlatformProfile(
            name="vendor-triton",
            device_types=("vendor",),
            backend_modes={"triton": frozenset({"jit"})},
        ),
    )

    target.validate_materialization("jit")

    with pytest.raises(ValueError, match="does not support `aot` materialization"):
        target.validate_materialization("aot")

    with pytest.raises(ValueError, match="capabilities for unsupported backends"):
        PlatformProfile(
            name="invalid-capability-backend",
            device_types=("vendor",),
            backend_unsupported_capabilities={"triton": frozenset({"dtype.fp8"})},
        )


def test_platform_registry_can_replace_profiles_without_api_changes():
    registry = PlatformRegistry()
    registry.register(
        PlatformProfile(
            name="vendor-card-v1",
            aliases=("vendor-current",),
            device_types=("vendor",),
        )
    )

    assert registry.get("vendor_current").name == "vendor-card-v1"

    registry.register(
        PlatformProfile(
            name="vendor-card-v1",
            aliases=("vendor-next",),
            device_types=("vendor",),
            compute_arch="arch-v2",
        ),
        replace=True,
    )

    assert registry.get("vendor-next").compute_arch == "arch-v2"

    with pytest.raises(ValueError, match="Unsupported platform"):
        registry.get("vendor-current")


def test_frozen_profile_mappings_cannot_change_target_identity():
    profile = PlatformProfile(
        name="immutable",
        constraints={"warp_size": 32, "schedule": {"stages": 2}},
        metadata={"source": "test"},
    )

    with pytest.raises(TypeError):
        profile.constraints["warp_size"] = 64

    with pytest.raises(TypeError):
        profile.metadata["source"] = "changed"

    with pytest.raises(TypeError):
        profile.constraints["schedule"]["stages"] = 3


def test_profile_metadata_is_canonical_and_rejects_mutable_leaf_values():
    profile = PlatformProfile(
        name="canonical",
        metadata={"features": {"zeta", "alpha"}},
    )

    assert profile.as_metadata()["metadata"]["features"] == ("alpha", "zeta")

    with pytest.raises(TypeError, match="Unsupported platform profile metadata value"):
        PlatformProfile(name="mutable-leaf", metadata={"payload": bytearray(b"a")})


def test_environment_target_resolution_remains_optional(monkeypatch):
    registry = _registry(
        PlatformProfile(
            name="environment-target",
            compute_arch="arch-v1",
            backend_modes={"triton": frozenset({"jit"})},
        )
    )
    monkeypatch.setenv("NINETOOTHED_PLATFORM", "environment-target")
    monkeypatch.setenv("NINETOOTHED_COMPUTE_ARCH", "arch-v1")

    target = resolve_target_context(None, registry=registry)

    assert target.backend == Target.TRITON
    assert target.platform.name == "environment-target"
    assert target.compute_arch == "arch-v1"


def _arrangement(input, other, output):
    return tuple(tensor.tile((64,)) for tensor in (input, other, output))


def _application(input, other, output):
    output = input + other  # noqa: F841


def _dot_arrangement(ninetoothed_dot_arg_0, rhs, output):
    output_tiled = output.tile((16, 16))
    lhs_tiled = (
        ninetoothed_dot_arg_0.tile((16, 16))
        .tile((1, -1))
        .expand((-1, output_tiled.shape[1]))
    )
    rhs_tiled = rhs.tile((16, 16)).tile((-1, 1)).expand((output_tiled.shape[0], -1))
    lhs_tiled.dtype = lhs_tiled.dtype.squeeze(0)
    rhs_tiled.dtype = rhs_tiled.dtype.squeeze(1)

    return lhs_tiled, rhs_tiled, output_tiled


def _dot_application(ninetoothed_dot_arg_0, rhs, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(ninetoothed_dot_arg_0.shape[0]):
        accumulator += ntl.dot(ninetoothed_dot_arg_0[k], rhs[k])

    output = accumulator.to(ntl.float16)


@pytest.fixture
def compiler_registry(monkeypatch):
    registry = PlatformRegistry()

    def resolve(backend, *, platform=None, compute_arch=None):
        return resolve_target_context(
            backend,
            platform=platform,
            compute_arch=compute_arch,
            registry=registry,
        )

    monkeypatch.setattr(
        "ninetoothed.compiler.driver.resolve_compile_target",
        resolve,
    )

    return registry


def test_native_cuda_profile_binds_vendor_toolchain(compiler_registry):
    profile = PlatformProfile(
        name="native-cuda",
        compute_arch="vendor-arch",
        backend_modes={"cuda": frozenset({"jit"})},
        metadata={"cuda": {"arch": "native", "wmma": False}},
    )
    compiler_registry.register(profile)
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=(Tensor(1), Tensor(1), Tensor(1)),
            backend="cuda",
            platform=profile.name,
        )
    )

    assert compilation.target.compute_arch == profile.compute_arch
    assert compilation.request.backend_options == {"arch": "native"}
    assert compilation.artifact.metadata["target"]["profile"]["metadata"]["cuda"] == {
        "arch": "native",
        "wmma": False,
    }

    with pytest.raises(ValueError, match="through its native vendor toolchain"):
        DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=_arrangement,
                application=_application,
                tensors=(Tensor(1), Tensor(1), Tensor(1)),
                backend="cuda",
                platform=profile.name,
                backend_options={"arch": "sm_90"},
            )
        )


def test_jit_only_triton_profile_is_rejected_before_aot(compiler_registry, tmp_path):
    profile = PlatformProfile(
        name="jit-only-aot-check",
        device_types=("vendor",),
        backend_modes={"triton": frozenset({"jit"})},
    )
    compiler_registry.register(profile)

    with pytest.raises(ValueError, match="does not support `aot` materialization"):
        DEFAULT_COMPILER.materialize(
            CompileRequest(
                arrangement=_arrangement,
                application=_application,
                tensors=(Tensor(1), Tensor(1), Tensor(1)),
                backend="triton",
                platform=profile.name,
            ),
            output_dir=tmp_path,
            mode="aot",
        )


def test_platform_profiles_apply_declared_compiler_constraints(compiler_registry):
    schedule_profile = PlatformProfile(
        name="bounded-schedule",
        backend_modes={"triton": frozenset({"jit"})},
        constraints={
            "compiler_options": {"triton": {"max_num_configs": 8, "num_stages": (1, 2)}}
        },
    )
    grid_profile = PlatformProfile(
        name="bounded-grid",
        backend_modes={"triton": frozenset({"jit"})},
        constraints={"compiler_options": {"triton": {"max_num_configs": 1}}},
        metadata={"triton_grid_limit": 65535},
    )
    plain_profile = PlatformProfile(
        name="plain-grid",
        backend_modes={"triton": frozenset({"jit"})},
    )
    block_profile = PlatformProfile(
        name="fixed-block",
        backend_modes={"triton": frozenset({"jit"})},
        constraints={"compiler_options": {"triton": {"max_num_configs": 1}}},
        metadata={"triton_block_size": 512},
    )

    for profile in (schedule_profile, grid_profile, plain_profile, block_profile):
        compiler_registry.register(profile)

    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=(Tensor(1), Tensor(1), Tensor(1)),
            backend="triton",
            platform=schedule_profile.name,
            max_num_configs=50,
        )
    )

    assert compilation.request.num_stages == (1, 2)
    assert compilation.request.max_num_configs == 8
    assert len(compilation.launch_plan.tuning_candidates) == 2

    explicit = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=(Tensor(1), Tensor(1), Tensor(1)),
            backend="triton",
            platform=schedule_profile.name,
            num_stages=3,
            max_num_configs=50,
        )
    )

    assert explicit.request.num_stages == 3
    assert explicit.request.max_num_configs == 8

    grid = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=(Tensor(1), Tensor(1), Tensor(1)),
            backend="triton",
            platform=grid_profile.name,
            max_num_configs=50,
        )
    )

    assert grid.request.max_num_configs == 1
    assert len(grid.launch_plan.tuning_candidates) == 1
    grid_source = grid.artifact.primary_source
    assert "tl.program_id(1)" not in grid_source
    assert "tl.range(tl.program_id(0)" in grid_source
    assert "tl.num_programs(0)" in grid_source
    assert "max(1, 65535 // _ninetoothed_num_warps)" in grid_source
    assert "if _ninetoothed_num_warps > 65535:" in grid_source

    unchanged = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=(Tensor(1), Tensor(1), Tensor(1)),
            backend="triton",
            platform=plain_profile.name,
            max_num_configs=50,
        )
    )
    unchanged_source = unchanged.artifact.primary_source
    assert "tl.range(tl.program_id(0)" not in unchanged_source
    assert "tl.num_programs(0)" not in unchanged_source
    assert "tl.program_id(1)" not in unchanged_source

    block = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=(Tensor(1), Tensor(1), Tensor(1)),
            backend="triton",
            platform=block_profile.name,
            max_num_configs=50,
        )
    )

    assert block.request.max_num_configs == 1
    assert len(block.launch_plan.tuning_candidates) == 1
    assert "block = 512" in block.artifact.primary_source


def test_dot_operand_coercion_is_driven_by_profile_metadata(compiler_registry):
    coerced_profile = PlatformProfile(
        name="coerced-dot",
        backend_modes={"triton": frozenset({"jit"})},
        constraints={"compiler_options": {"triton": {"fixed_num_stages": 1}}},
        metadata={"triton_dot_operand_coercions": {"float32": "float16"}},
    )
    baseline_profile = PlatformProfile(
        name="baseline-dot",
        backend_modes={"triton": frozenset({"jit"})},
    )
    invalid_profile = PlatformProfile(
        name="invalid-dot-coercion",
        backend_modes={"triton": frozenset({"jit"})},
        metadata={"triton_dot_operand_coercions": {"float32": None}},
    )
    compiler_registry.register(coerced_profile)
    compiler_registry.register(baseline_profile)
    compiler_registry.register(invalid_profile)

    def compile_for(profile, *, dtype="float32", **options):
        tensor_dtypes = (
            None
            if dtype is None
            else {name: dtype for name in ("ninetoothed_dot_arg_0", "rhs", "output")}
        )

        return DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=_dot_arrangement,
                application=_dot_application,
                tensors=(
                    Tensor(shape=(Symbol("ninetoothed_dot_arg_1"), None)),
                    Tensor(2),
                    Tensor(2),
                ),
                backend="triton",
                platform=profile.name,
                max_num_configs=1,
                tensor_dtypes=tensor_dtypes,
                **options,
            )
        )

    coerced = compile_for(coerced_profile)
    coerced_runtime_dtype = compile_for(coerced_profile, dtype=None)
    baseline = compile_for(baseline_profile)
    coerced_bfloat16 = compile_for(coerced_profile, dtype="bfloat16")
    coerced_dot_lines = tuple(
        line
        for line in coerced.artifact.primary_source.splitlines()
        if "tl.dot(" in line
    )
    coerced_runtime_dot_lines = tuple(
        line
        for line in coerced_runtime_dtype.artifact.primary_source.splitlines()
        if "tl.dot(" in line
    )
    baseline_dot_lines = tuple(
        line
        for line in baseline.artifact.primary_source.splitlines()
        if "tl.dot(" in line
    )
    coerced_bfloat16_dot_lines = tuple(
        line
        for line in coerced_bfloat16.artifact.primary_source.splitlines()
        if "tl.dot(" in line
    )

    assert coerced.request.num_stages == 1
    assert baseline.request.num_stages is None
    assert coerced_dot_lines
    assert all(line.count(".to(tl.float16)") == 2 for line in coerced_dot_lines)
    assert all(line.count(".to(tl.float16)") == 2 for line in coerced_runtime_dot_lines)
    assert all(
        line.count(".dtype == tl.float32") == 2 for line in coerced_runtime_dot_lines
    )
    runtime_source = coerced_runtime_dtype.artifact.primary_source
    assert "ninetoothed_dot_arg_0 = " not in runtime_source
    assert "ninetoothed_dot_arg_1 = " not in runtime_source
    assert "ninetoothed_dot_arg_2 = " in runtime_source
    assert any(".to(tl.float16)" not in line for line in baseline_dot_lines)
    assert coerced_bfloat16_dot_lines
    assert all(".to(tl.float16)" not in line for line in coerced_bfloat16_dot_lines)

    with pytest.raises(ValueError, match="requires dtype names"):
        compile_for(invalid_profile)

    with pytest.raises(ValueError, match="requires num_stages=1"):
        compile_for(coerced_profile, num_stages=2)
