import json
from pathlib import Path

import pytest
import torch

import ninetoothed
from ninetoothed import Tensor, block_size
from ninetoothed.backends.materializers.triton import _aot_wrapper
from ninetoothed.compiler import load_built_artifact
from ninetoothed.compiler.layout_runtime import build_layout_transfer_validator
from ninetoothed.ir import LaunchABI, LaunchBinding
from tests.utils import get_available_devices, requires_effective_backend

BLOCK_SIZE = block_size(16, 32)


def _permutation_arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return (
        input.tile((BLOCK_SIZE, BLOCK_SIZE)),
        output.permute((1, 0)).tile((BLOCK_SIZE, BLOCK_SIZE)),
    )


def _copy_application(input, output):
    output = input  # noqa: F841


def _abi():
    return LaunchABI(
        public_args=("x", "out"),
        kernel_args=(
            LaunchBinding(name="x", kind="tensor", source="x"),
            LaunchBinding(name="out", kind="tensor", source="out"),
            LaunchBinding(name="sx0", kind="shape", source="x", dim=0),
            LaunchBinding(name="sx1", kind="shape", source="x", dim=1),
            LaunchBinding(name="dy0", kind="shape", source="out", dim=0),
            LaunchBinding(name="dy1", kind="shape", source="out", dim=1),
        ),
        outputs=("out",),
    )


def _metadata(**overrides):
    transfer = {
        "source_binding": "x",
        "destination_binding": "out",
        "permutation": (1, 0),
        "physical_constraints": (("sx0", "dy1"), ("sx1", "dy0")),
        "value_constraints": ((("sx1", "sx0"), ("dy0", "dy1")),),
        "program_constraints": ((("sx1", "sx0"), ("dy0", "dy1")),),
    }
    transfer.update(overrides)

    return {"layout_transfer": transfer}


def test_layout_transfer_runtime_contract_and_aot_wiring():
    abi = _abi()
    metadata = _metadata()
    x = torch.empty((2, 3))
    out = torch.empty((3, 2))
    validate = build_layout_transfer_validator(metadata, abi)

    validate({"x": x, "out": out})

    provenance_only = _metadata(
        physical_constraints=(),
        value_constraints=(),
        program_constraints=(),
    )

    with pytest.raises(ValueError, match="physical shapes"):
        build_layout_transfer_validator(provenance_only, abi)(
            {"x": x, "out": torch.empty((2, 3))}
        )

    with pytest.raises(ValueError, match="destination strides"):
        validate({"x": x, "out": torch.empty_strided((3, 2), (0, 1))})

    validate({"x": x, "out": torch.empty_strided((3, 2), (3, 4))})

    for constraint, replacement, expected in (
        ("physical_constraints", (("sx0", "dy0"),), "physical shapes"),
        (
            "value_constraints",
            ((("sx0",), ("dy0",)),),
            "value domains",
        ),
        (
            "program_constraints",
            ((("sx0",), ("dy0",)),),
            "program domains",
        ),
    ):
        with pytest.raises(ValueError, match=expected):
            build_layout_transfer_validator(
                _metadata(**{constraint: replacement}), abi
            )({"x": x, "out": out})

    storage = torch.empty(7)
    overlapping_x = storage[:6].view(2, 3)
    overlapping_out = storage[1:7].view(3, 2)

    with pytest.raises(ValueError, match="non-overlapping"):
        validate({"x": overlapping_x, "out": overlapping_out})

    class Function:
        argtypes = None
        called = False

        def __call__(self, *_args):
            self.called = True

            return 0

    function = Function()
    wrapped = _aot_wrapper(
        function,
        lambda: 0,
        lambda: None,
        abi,
        (),
        validate_bindings=build_layout_transfer_validator(
            _metadata(program_constraints=((("sx0",), ("dy0",)),)), abi
        ),
    )

    with pytest.raises(ValueError, match="program domains"):
        wrapped(x, out)

    assert not function.called

    empty_wrapped = _aot_wrapper(
        function,
        lambda: 0,
        lambda: None,
        abi,
        (),
        validate_bindings=validate,
    )

    with pytest.raises(ValueError, match="physical shapes"):
        empty_wrapped(torch.empty((0, 3)), torch.empty((0, 2)))

    assert not function.called


@pytest.mark.parametrize("device", get_available_devices())
@requires_effective_backend("triton")
def test_layout_transfer_jit_aot_reload_with_dynamic_strides(device, tmp_path):
    rows, columns = 127, 79
    input = torch.empty_strided(
        (rows, columns),
        (columns + 7, 1),
        dtype=torch.float16,
        device=device,
    )
    input.copy_(torch.randn_like(input))
    output = torch.empty_strided(
        (columns, rows),
        (rows + 5, 1),
        dtype=torch.float16,
        device=device,
    )
    jit_kernel = ninetoothed.make(
        _permutation_arrangement,
        _copy_application,
        (Tensor(2), Tensor(2)),
        backend="triton",
    )

    jit_kernel(input, output)
    torch.testing.assert_close(output, input.T, rtol=0, atol=0)

    shared = torch.randn((64, 64), dtype=torch.float16, device=device)

    with pytest.raises(ValueError, match="non-overlapping"):
        jit_kernel(shared, shared.T)

    aot_kernel = ninetoothed.make(
        _permutation_arrangement,
        _copy_application,
        (
            Tensor(2, dtype=ninetoothed.float16),
            Tensor(2, dtype=ninetoothed.float16),
        ),
        backend="triton",
        caller=device,
        output_dir=tmp_path,
    )
    manifest = json.loads(
        Path(aot_kernel._built_artifact.manifest_path).read_text(encoding="utf-8")
    )

    assert len(manifest["launch_plan"]["tuning_candidates"]) == 1

    for launch in (aot_kernel, load_built_artifact(aot_kernel._built_artifact)):
        output.zero_()
        launch(input, output)
        torch.testing.assert_close(output, input.T, rtol=0, atol=0)

        with pytest.raises(ValueError, match="non-overlapping"):
            launch(shared, shared.T)
