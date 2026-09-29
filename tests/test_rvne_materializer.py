import os
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ninetoothed import Tensor
from ninetoothed.backends.core import Artifact, BuiltArtifact, Target
from ninetoothed.backends.materializers import rvne
from ninetoothed.backends.rvne_toolchain import (
    find_rvne_toolchain,
    rvne_compile_command,
    rvne_compiler_identity,
)
from ninetoothed.ir import LaunchABI, LaunchBinding, ir_to_dict


def _aot_affine(
    x: Tensor(shape=(4,), dtype="i32"), out: Tensor(shape=(4,), dtype="i32")
):
    out = x * 3 + 7  # noqa: F841


def _contract():
    abi = LaunchABI(
        public_args=("x", "out"),
        kernel_args=(
            LaunchBinding(name="x", source="x", kind="tensor", access="read"),
            LaunchBinding(name="out", source="out", kind="tensor", access="write"),
            LaunchBinding(name="n", source="x", kind="shape", dim=0),
        ),
        outputs=("out",),
    )
    specs = {
        name: {"name": name, "dtype": "int32", "ndim": 1, "shape": ("4",), "attrs": {}}
        for name in ("x", "out")
    }

    return abi, specs


def _fake_toolchain(tmp_path):
    return SimpleNamespace(
        root=tmp_path,
        compiler=tmp_path / "clang++",
        emulator=tmp_path / "qemu-riscv64",
        gcc_toolchain=tmp_path / "gcc",
        sysroot=tmp_path / "sysroot",
    )


@pytest.mark.parametrize(
    "dtype,value,error",
    (
        ("int32", np.float32(1.9), TypeError),
        ("int32", np.int64(1), TypeError),
        ("float32", np.float64(1.0), TypeError),
        ("int32", 1.0, TypeError),
        ("int32", True, TypeError),
        ("bool", 1, TypeError),
        ("int32", 2**31, OverflowError),
        ("int32", -(2**31) - 1, OverflowError),
        ("uint32", -1, OverflowError),
        ("float32", 1.0e40, OverflowError),
        ("float32", 10**400, OverflowError),
        ("int32", np.array([1], dtype=np.int32), TypeError),
        ("int32", "1", TypeError),
    ),
)
def test_rvne_scalar_contract_rejects_casts_before_execution(
    monkeypatch, tmp_path, dtype, value, error
):
    abi = LaunchABI(
        public_args=("value",),
        kernel_args=(LaunchBinding(name="value", source="value", kind="scalar"),),
    )
    specs = {
        "value": {"name": "value", "dtype": dtype, "ndim": 0, "shape": (), "attrs": {}}
    }

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid scalar reached QEMU.")

    monkeypatch.setattr(rvne.subprocess, "run", forbidden)
    launch = rvne._wrapper(
        tmp_path / "kernel.elf", abi, specs, _fake_toolchain(tmp_path)
    )

    with pytest.raises(error):
        launch(value)


@pytest.mark.parametrize(
    "dtype,value",
    (
        ("int32", -(2**31)),
        ("uint32", 2**32 - 1),
        ("int64", 2**63 - 1),
        ("int32", np.int32(7)),
        ("int32", np.array(7, dtype=np.int32)),
        ("float32", np.float32(1.9)),
        ("float32", 1.9),
        ("float32", 7),
        ("bool", True),
        ("bool", np.bool_(False)),
    ),
)
def test_rvne_scalar_contract_preserves_supported_values(dtype, value):
    actual = rvne._scalar_value(value, np.dtype(dtype), "value")
    assert actual.dtype == np.dtype(dtype)
    assert actual.shape == ()
    assert actual.tobytes() == np.asarray(value, dtype=dtype).tobytes()


def test_rvne_wrapper_transports_arrays_and_shape_without_loading_library(
    monkeypatch, tmp_path
):
    abi, specs = _contract()
    x = np.arange(4, dtype=np.int32)
    out = np.full(4, -1, dtype=np.int32)
    observed = []

    def run(command, **options):
        assert options["check"] and options["timeout"] == 120
        observed.append(command)
        input_path, output_path, shape_path = map(Path, command[4:])
        assert input_path.read_bytes() == struct.pack("<Q", 4) + x.tobytes()
        assert shape_path.read_bytes() == struct.pack("<q", 4)
        output_path.write_bytes(struct.pack("<Q", 4) + (x + 7).tobytes())

        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(rvne.subprocess, "run", run)
    launch = rvne._wrapper(
        tmp_path / "kernel.elf", abi, specs, _fake_toolchain(tmp_path)
    )
    assert launch(x, out) is out
    np.testing.assert_array_equal(out, x + 7)
    assert len(observed) == 1
    assert not Path(observed[0][4]).exists()


@pytest.mark.parametrize("invalid", ("dtype", "shape", "stride", "readonly", "alias"))
def test_rvne_rejects_incompatible_arrays_before_execution(
    monkeypatch, tmp_path, invalid
):
    abi, specs = _contract()
    x = np.arange(4, dtype=np.int32)
    out = np.zeros(4, dtype=np.int32)

    if invalid == "dtype":
        x = x.astype(np.float32)
    elif invalid == "shape":
        x = np.zeros(5, dtype=np.int32)
    elif invalid == "stride":
        x = np.arange(8, dtype=np.int32)[::2]
    elif invalid == "readonly":
        out.flags.writeable = False
    elif invalid == "alias":
        out = x

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid arguments reached QEMU.")

    monkeypatch.setattr(rvne.subprocess, "run", forbidden)
    launch = rvne._wrapper(
        tmp_path / "kernel.elf", abi, specs, _fake_toolchain(tmp_path)
    )

    with pytest.raises((TypeError, ValueError)):
        launch(x, out)


def test_rvne_emulator_failure_preserves_host_output(monkeypatch, tmp_path):
    abi, specs = _contract()
    x = np.arange(4, dtype=np.int32)
    out = np.full(4, 99, dtype=np.int32)

    def fail(command, **kwargs):
        Path(command[5]).write_bytes(b"partial output")
        raise subprocess.CalledProcessError(1, command, stderr="illegal instruction")

    monkeypatch.setattr(rvne.subprocess, "run", fail)
    launch = rvne._wrapper(
        tmp_path / "kernel.elf", abi, specs, _fake_toolchain(tmp_path)
    )

    with pytest.raises(RuntimeError, match="illegal instruction"):
        launch(x, out)

    np.testing.assert_array_equal(out, np.full(4, 99, dtype=np.int32))


def test_rvne_compiler_failure_does_not_publish_partial_binary(monkeypatch, tmp_path):
    binary = tmp_path / "kernel.elf"

    def fail(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial executable")
        raise subprocess.CalledProcessError(
            1, command, stderr="unsupported instruction"
        )

    monkeypatch.setattr(rvne.subprocess, "run", fail)

    with pytest.raises(RuntimeError, match="unsupported instruction"):
        rvne._compile_executable(
            _fake_toolchain(tmp_path), tmp_path / "kernel.cpp", binary
        )

    assert not binary.exists()


def test_rvne_cross_compile_flags_preserve_target_and_integer_semantics(tmp_path):
    command = rvne_compile_command(_fake_toolchain(tmp_path), "in.cpp", "out.elf")
    assert "--target=riscv64-unknown-linux-gnu" in command
    assert "-march=rv64imafcvzne" in command
    assert "-fwrapv" in command
    assert "-static" in command
    assert "-shared" not in command


def test_rvne_real_sdk_harness_and_reload(tmp_path):
    if not os.environ.get("NINETOOTHED_RVNE_TOOLCHAIN"):
        pytest.skip("RVNE SDK is not configured")

    toolchain = find_rvne_toolchain()
    abi, specs = _contract()
    source_text = """
#include <stdint.h>
extern "C" int launch_add(const int32_t *x, int32_t *out, int64_t n) {
  for (int64_t i = 0; i < n; ++i) out[i] = x[i] + 7;
  return 0;
}
"""
    source = tmp_path / "add.cpp"
    source.write_text(
        source_text + rvne._harness("launch_add", abi, specs), encoding="utf-8"
    )
    binary = tmp_path / "add.elf"
    rvne._compile_executable(toolchain, source, binary)
    artifact = Artifact(
        backend=Target.RVNE,
        kernel_name="add",
        language="c++",
        sources={"add.cpp": source_text},
        entrypoint="launch_add",
        metadata={
            "tensors": tuple(specs.values()),
            "rvne_runtime": {"toolchain_root": str(toolchain.root), "protocol": 1},
        },
    )
    built = BuiltArtifact(
        source=artifact,
        cache_key="test",
        source_path=str(source),
        binary_path=str(binary),
        manifest_path=str(tmp_path / "manifest.json"),
        abi=ir_to_dict(abi),
    )
    reloaded = rvne.RvneMaterializer().load_built_artifact(built)
    x = np.array([0, -8, 11, np.iinfo(np.int32).max], dtype=np.int32)
    out = np.empty_like(x)
    assert reloaded(x, out) is out
    np.testing.assert_array_equal(out, x + np.int32(7))
    identity = rvne_compiler_identity(required=True)
    assert identity["available"]
    assert identity["isa"] == "rv64imafcvzne"
    assert identity["version"]


def test_rvne_real_aot_compiler_and_published_artifact_reload(tmp_path):
    if not os.environ.get("NINETOOTHED_RVNE_TOOLCHAIN"):
        pytest.skip("RVNE SDK is not configured")

    from ninetoothed.compiler import aot
    from ninetoothed.compiler.runtime import load_built_artifact

    handle = aot(
        _aot_affine, backend="rvne", kernel_name="rvne_affine", output_dir=tmp_path
    )
    x = np.array([-19, 0, 2, np.iinfo(np.int32).max], dtype=np.int32)
    out = np.empty_like(x)
    assert handle(x, out) is out
    np.testing.assert_array_equal(out, x * np.int32(3) + np.int32(7))
    built = handle._built_artifact
    assert Path(built.source_path).parent == tmp_path
    assert Path(built.binary_path).parent == tmp_path
    assert Path(built.manifest_path).is_file()
    loaded = load_built_artifact(built)
    out.fill(0)
    loaded(x, out)
    np.testing.assert_array_equal(out, x * np.int32(3) + np.int32(7))
