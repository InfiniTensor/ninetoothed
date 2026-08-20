import pytest
import torch

import ninetoothed
from ninetoothed import Tensor
from ninetoothed.backends import Target, emit
from ninetoothed.frontend.layout import tensor_specs
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import Kernel, TensorSpec
from ninetoothed.targets import (
    PlatformProfile,
    PlatformRegistry,
    TargetContext,
    resolve_target_context,
)
from tests.utils import get_available_devices


def _transpose_kernel(*, compiler_options=None) -> Kernel:
    source = "\ndef arbitrary_name(x, out):\n    out = x.T\n"
    tensors = (
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="x"),
        TensorSpec(ndim=2, shape=("n", "m"), dtype="float32", name="out"),
    )
    program = from_source(source, tensors, kind="layout_transfer_test")
    assert program is not None

    return Kernel(
        kernel_name="layout_transfer_test",
        source=source,
        source_language="ninetoothed-python",
        entrypoint="layout_transfer_test",
        tensors=tensors,
        compiler_options=compiler_options or {},
        ssa=program,
    )


def test_triton_layout_transfer_emits_tiled_transpose_with_independent_tails():
    artifact = emit(_transpose_kernel(), Target.TRITON)
    source = artifact.primary_source

    compile(source, "layout_transfer_test.triton.py", "exec")

    for fragment in (
        "TILE_M: tl.constexpr",
        "TILE_N: tl.constexpr",
        "_ninetoothed_tile_m=16",
        "_ninetoothed_tile_n=16",
        "grid = (((n + _ninetoothed_tile_m - 1) // _ninetoothed_tile_m) * "
        "((m + _ninetoothed_tile_n - 1) // _ninetoothed_tile_n),)",
        "source_value_0 = transfer_tile_row * TILE_M + tl.arange(0, TILE_M)[None, :]",
        "source_value_1 = transfer_tile_column * TILE_N + "
        "tl.arange(0, TILE_N)[:, None]",
        "transfer_value = tl.trans(transfer_value)",
    ):
        assert fragment in source

    assert "triton.cdiv(n, _ninetoothed_tile_m)" not in source

    load = next(
        line for line in source.splitlines() if "transfer_value = tl.load" in line
    )
    store = next(line for line in source.splitlines() if "tl.store(out +" in line)
    assert "source_value_0 < (n)" in load
    assert "source_value_1 < (m)" in load
    assert "destination_value_0" not in load
    assert "destination_value_0 < (n)" in store
    assert "destination_value_1 < (m)" in store
    assert "source_value_0" not in store


def test_triton_layout_transfer_exposes_legal_schedule_candidates():
    artifact = emit(_transpose_kernel(), Target.TRITON)
    candidates = artifact.metadata["ssa_schedule_candidates"]

    assert tuple(candidate["name"] for candidate in candidates) == (
        "transpose-16x16",
        "transpose-32x32",
    )
    assert tuple(
        (
            candidate["schedule"]["tile"],
            candidate["schedule"]["num_warps"],
            candidate["schedule"]["num_stages"],
        )
        for candidate in candidates
    ) == (
        ({"block_m": 16, "block_n": 16}, 4, 1),
        ({"block_m": 32, "block_n": 32}, 8, 1),
    )
    assert artifact.metadata["ssa_schedule"]["tile"] == {
        "block_m": 16,
        "block_n": 16,
    }
    assert artifact.metadata["layout_transfer"]["private_meta_parameters"] == (
        "TILE_M",
        "TILE_N",
    )
    assert artifact.metadata["layout_transfer"]["value_constraints"] == (
        (("n", "m"), ("n", "m")),
    )
    assert artifact.metadata["layout_transfer"]["physical_constraints"] == (
        ("n", "n"),
        ("m", "m"),
    )
    assert artifact.metadata["layout_transfer"]["program_constraints"] == (
        (("n", "m"), ("n", "m")),
    )


def test_explicit_candidate_survives_single_config_limit_with_selected_warps():
    artifact = emit(
        _transpose_kernel(
            compiler_options={
                "max_num_configs": 1,
                "ssa_pass_options": {
                    "ssa.triton.optimize_schedule": {"candidate": "transpose-32x32"}
                },
            }
        ),
        Target.TRITON,
    )

    (candidate,) = artifact.metadata["ssa_schedule_candidates"]
    assert candidate["name"] == "transpose-32x32"
    assert candidate["schedule"] == {
        "tile": {"block_m": 32, "block_n": 32},
        "num_warps": 8,
        "num_stages": 1,
    }
    assert artifact.metadata["ssa_schedule"]["num_warps"] == 8
    assert "_ninetoothed_num_warps=8" in artifact.primary_source


def test_non_power_of_two_pre_tiled_transfer_uses_generic_emission():
    tensors = tensor_specs(
        ("x", "out"),
        (
            Tensor(shape=(8, 12)).permute((1, 0)).tile((3, 2)),
            Tensor(shape=(12, 8)).tile((3, 2)),
        ),
    )
    source = "\ndef copy(x, out):\n    out = x\n"
    program = from_source(source, tensors, kind="layout_transfer_test")
    assert program is not None
    artifact = emit(
        Kernel(
            kernel_name="layout_transfer_test",
            source=source,
            source_language="ninetoothed-python",
            entrypoint="layout_transfer_test",
            tensors=tensors,
            ssa=program,
        ),
        Target.TRITON,
    )

    assert "transfer_value = tl.trans(transfer_value)" not in artifact.primary_source
    assert "tl.arange(0, 3)" not in artifact.primary_source
    assert "layout_transfer" not in artifact.metadata


def test_layout_transfer_grid_stride_preserves_unbounded_source():
    baseline = emit(_transpose_kernel(), Target.TRITON).primary_source
    unbounded_profile = PlatformProfile(
        name="synthetic-unbounded",
        backend_modes={"triton": frozenset({"jit"})},
    )
    unbounded = emit(
        _transpose_kernel(),
        Target.TRITON,
        target_context=TargetContext(
            backend=Target.TRITON,
            platform=unbounded_profile,
        ),
    ).primary_source
    bounded_profile = PlatformProfile(
        name="synthetic-grid-limit",
        backend_modes={"triton": frozenset({"jit"})},
        metadata={"triton_grid_limit": 65535},
    )
    bounded = emit(
        _transpose_kernel(),
        Target.TRITON,
        target_context=TargetContext(
            backend=Target.TRITON,
            platform=bounded_profile,
        ),
    ).primary_source

    assert unbounded == baseline
    assert "tl.program_id(1)" not in bounded
    assert "tl.range(tl.program_id(0)" in bounded
    assert "tl.num_programs(0)" in bounded
    assert "transfer_value = tl.trans(transfer_value)" not in bounded
    assert "tl.arange(0, TILE_M)[:, None]" in bounded
    assert "tl.arange(0, TILE_N)[None, :]" in bounded
    assert "max(1, 65535 // _ninetoothed_num_warps)" in bounded


@pytest.mark.parametrize("device", get_available_devices())
def test_layout_transfer_grid_stride_runs_on_accelerator(device, monkeypatch):
    profile = PlatformProfile(
        name="synthetic-layout-grid-stride",
        device_types=(device,),
        backend_modes={"triton": frozenset({"jit"})},
        metadata={"triton_grid_limit": 64},
    )
    registry = PlatformRegistry()
    registry.register(profile)
    monkeypatch.setattr(
        "ninetoothed.compiler.driver.resolve_compile_target",
        lambda backend, *, platform=None, compute_arch=None: resolve_target_context(
            backend,
            platform=platform,
            compute_arch=compute_arch,
            registry=registry,
        ),
    )

    def arrangement(input, output):
        return input.tile((16, 32)), output.permute((1, 0)).tile((16, 32))

    def application(input, output):
        output = input  # noqa: F841

    rows = columns = 1024
    input = torch.arange(rows * columns, dtype=torch.float32, device=device).reshape(
        rows,
        columns,
    )
    output = torch.empty((columns, rows), dtype=input.dtype, device=device)
    kernel = ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(2)),
        backend="triton",
        platform=profile.name,
        num_warps=4,
        max_num_configs=1,
    )

    kernel(input, output)

    torch.testing.assert_close(output, input.T, rtol=0, atol=0)
