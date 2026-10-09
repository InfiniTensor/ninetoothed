import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from examples.rvne.operators import RvneOperators
from ninetoothed.backends.emitters.rvne import emit
from ninetoothed.backends.rvne_toolchain import find_rvne_toolchain
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import Kernel, TensorSpec


@pytest.fixture(scope="module")
def rvne_sdk():
    if not os.environ.get("NINETOOTHED_RVNE_TOOLCHAIN"):
        pytest.skip("RVNE SDK is not configured")

    return find_rvne_toolchain()


def _assert_riscv_artifacts(operators):
    assert operators.artifact_paths

    for filename in operators.artifact_paths:
        path = Path(filename)
        assert path.suffix == ".elf"
        header = path.read_bytes()[:20]
        assert header[:6] == b"\x7fELF\x02\x01"
        assert int.from_bytes(header[18:20], "little") == 243


def test_rvne_model_rejects_incompatible_spike_broadcast():
    source = (
        "def application(current, spikes, low, high, out):\n"
        "    out = ntl.spike_accumulate(current, spikes, low, high)\n"
    )
    tensors = (
        TensorSpec(ndim=2, shape=("2", "3"), dtype="int32", name="current"),
        TensorSpec(ndim=2, shape=("2", "2"), dtype="uint32", name="spikes"),
        TensorSpec(ndim=2, shape=("2", "3"), dtype="uint64", name="low"),
        TensorSpec(ndim=2, shape=("2", "3"), dtype="uint64", name="high"),
        TensorSpec(ndim=2, shape=("2", "3"), dtype="int32", name="out"),
    )
    program = from_source(source, tensors, kind="invalid_spike_broadcast")
    assert program is not None
    kernel = Kernel(
        kernel_name="invalid_spike_broadcast",
        source=source,
        source_language="ninetoothed-python",
        tensors=tensors,
        ssa=program,
    )

    with pytest.raises(ValueError, match="operands must broadcast"):
        emit(kernel)


@pytest.mark.parametrize(
    "input_shape,weight_shape,stride,padding",
    (
        ((2, 7, 6, 7), (3, 7, 3, 2), (2, 1), (1, 0)),
        ((1, 115, 4, 5), (2, 115, 3, 3), (1, 1), (1, 1)),
    ),
    ids=("nonsquare-multiple-batches", "1035-synapses"),
)
def test_rvne_model_convolution_qemu_without_spatial_weight_replication(
    tmp_path, rvne_sdk, input_shape, weight_shape, stride, padding
):
    operators = RvneOperators(tmp_path, toolchain_root=str(rvne_sdk.root))
    rng = np.random.default_rng(709 + input_shape[1])
    weights = rng.integers(-8, 8, weight_shape, dtype=np.int32)
    weights.flat[0] = -8
    weights.flat[-1] = 7
    prepared = operators.prepare_weights(weights)
    blocks = (int(np.prod(weight_shape[1:])) + 31) // 32
    expected_weight_bytes = weight_shape[0] * blocks * 16
    assert prepared.shape == weight_shape
    assert prepared.low.shape == (blocks, weight_shape[0])
    assert prepared.high.shape == prepared.low.shape
    assert prepared.packed_nbytes == expected_weight_bytes
    low, high = prepared.low.copy(), prepared.high.copy()

    for extra_rows in (0, 1):
        shape = (*input_shape[:-2], input_shape[-2] + extra_rows, input_shape[-1])
        inputs = rng.integers(0, 2, shape, dtype=np.int32)
        expected = (
            functional.conv2d(
                torch.from_numpy(inputs).float(),
                torch.from_numpy(weights).float(),
                stride=stride,
                padding=padding,
            )
            .numpy()
            .astype(np.int32)
        )
        actual = operators.conv2d(inputs, prepared, stride=stride, padding=padding)
        assert actual.dtype == np.dtype("int32")
        np.testing.assert_array_equal(actual, expected)
        positions = shape[0] * expected.shape[-2] * expected.shape[-1]
        staging = operators.last_staging
        assert staging["weights_bytes"] == expected_weight_bytes
        assert staging["patches_bytes"] == positions * blocks * 4
        assert staging["output_bytes"] == expected.nbytes
        assert prepared.packed_nbytes == expected_weight_bytes
        np.testing.assert_array_equal(prepared.low, low)
        np.testing.assert_array_equal(prepared.high, high)

    _assert_riscv_artifacts(operators)


def test_rvne_model_max_pool_qemu_discards_odd_edges_and_preserves_negative_values(
    tmp_path, rvne_sdk
):
    operators = RvneOperators(tmp_path, toolchain_root=str(rvne_sdk.root))
    rng = np.random.default_rng(719)
    inputs = rng.integers(-100, 0, (2, 3, 5, 7), dtype=np.int32)
    inputs[:, :, -1, :] = 123
    inputs[:, :, :, -1] = 456
    expected = (
        functional.max_pool2d(torch.from_numpy(inputs).float(), kernel_size=2, stride=2)
        .numpy()
        .astype(np.int32)
    )
    actual = operators.max_pool2d(inputs)
    assert actual.shape == (2, 3, 2, 3)
    assert actual.dtype == np.dtype("int32")
    assert np.all(expected < 0)
    np.testing.assert_array_equal(actual, expected)
    _assert_riscv_artifacts(operators)


def test_rvne_model_lif_qemu_preserves_state_across_timesteps(tmp_path, rvne_sdk):
    operators = RvneOperators(tmp_path, toolchain_root=str(rvne_sdk.root))
    rng = np.random.default_rng(727)
    voltage = rng.integers(-20, 21, 33, dtype=np.int32)
    syn_current = rng.integers(-30, 31, 33, dtype=np.int32)
    voltage[:4] = (1, 2, 0, -1)
    syn_current[:4] = 0

    for timestep in range(5):
        current = rng.integers(-45, 46, 33, dtype=np.int32)

        if timestep == 0:
            current[:4] = (0, 0, 8, 0)

        integrated = syn_current + current
        potential = voltage - (voltage >> 10) + (integrated >> 3)
        decayed = integrated - (integrated >> 4)
        expected_spike = (potential > 1).astype(np.int32)
        expected_voltage = np.where(expected_spike != 0, 0, potential)
        expected_current = np.where(expected_spike != 0, 0, decayed)
        old_voltage, old_current = voltage.copy(), syn_current.copy()
        spike, next_voltage, next_current = operators.lif(current, voltage, syn_current)
        assert all(
            value.dtype == np.dtype("int32")
            for value in (spike, next_voltage, next_current)
        )
        np.testing.assert_array_equal(spike, expected_spike)
        np.testing.assert_array_equal(next_voltage, expected_voltage)
        np.testing.assert_array_equal(next_current, expected_current)
        np.testing.assert_array_equal(voltage, old_voltage)
        np.testing.assert_array_equal(syn_current, old_current)
        voltage, syn_current = next_voltage, next_current

    _assert_riscv_artifacts(operators)


def test_rvne_sconv_lif_model_qemu_matches_every_stage(tmp_path, rvne_sdk):
    model_dir = os.environ.get("NINETOOTHED_SCONVLIF_MODEL_DIR")

    if not model_dir:
        pytest.skip("external SConvLif model is not configured")

    from examples.rvne.sconv_lif import verify_model

    report = verify_model(
        model_dir,
        tmp_path,
        toolchain_root=str(rvne_sdk.root),
        seed=20260830,
        verbose=False,
    )
    assert report["status"] == "passed"
    assert report["output_shape"] == [1, 3, 2, 2, 9]
    assert report["final_mismatches"] == 0
    assert report["final_nonzero"] == 108
    assert report["comparison_count"] == 165
    assert report["launch_count"] == 88
    assert report["launch_counts"] == {"conv2d": 40, "max_pool2d": 12, "lif": 36}
