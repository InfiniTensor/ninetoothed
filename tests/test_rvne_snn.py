import os

import numpy as np
import pytest

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import CompileRequest, compile_kernel
from ninetoothed.compiler.runtime import materialize


def _identity(*tensors):
    return tensors


def _quantized_matmul(current, spikes, weights, out):
    result = spikes @ weights
    out = current + result  # noqa: F841


def _spike_escape_hatch(current, spikes, low, high, out):
    out = ntl.spike_accumulate(current, spikes, low, high)  # noqa: F841


def _lif_application(current, voltage, syn_current, spike, next_voltage, next_current):
    integrated = syn_current + current
    potential = voltage - (voltage >> 10)
    potential = potential + (integrated >> 3)
    decayed = integrated - (integrated >> 4)
    fired = potential > 1
    spike = ntl.where(fired, 1, 0)  # noqa: F841
    next_voltage = ntl.where(fired, 0, potential)  # noqa: F841
    next_current = ntl.where(fired, 0, decayed)  # noqa: F841


def _compile(application, arrays):
    return compile_kernel(
        CompileRequest(
            arrangement=_identity,
            application=application,
            tensors=tuple(
                Tensor(
                    shape=a.shape,
                    dtype="int4" if a.dtype == np.int8 else a.dtype.name,
                )
                for a in arrays
            ),
            backend="rvne",
            caller="numpy",
        )
    )


def _require_sdk():
    if not os.environ.get("NINETOOTHED_RVNE_TOOLCHAIN"):
        pytest.skip("RVNE SDK is not configured")


def test_rvne_lowers_standard_bool_int4_dot_to_spike_instruction():
    current = np.zeros((2, 3), dtype=np.int32)
    spikes = np.array(
        [
            [0, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
        ],
        dtype=bool,
    )
    weights = np.arange(-2, 13, dtype=np.int8).reshape(5, 3)
    out = np.empty_like(current)
    compilation = _compile(_quantized_matmul, (current, spikes, weights, out))

    assert "__builtin_riscv_calc_acc_32" in compilation.artifact.primary_source


def test_rvne_rejects_frontend_spike_escape_hatch():
    arrays = (
        np.zeros(3, dtype=np.int32),
        np.zeros(3, dtype=np.uint32),
        np.zeros(3, dtype=np.uint64),
        np.zeros(3, dtype=np.uint64),
        np.zeros(3, dtype=np.int32),
    )

    with pytest.raises(ValueError, match="call.spike_accumulate"):
        _compile(_spike_escape_hatch, arrays)


def _reversed_int4_matmul(current, weights, spikes, out):
    result = weights @ spikes
    out = current + result  # noqa: F841


def _int4_arithmetic(weights, out):
    out = weights + weights  # noqa: F841


def test_rvne_rejects_reversed_and_non_contracted_int4_usage():
    current = np.zeros((3, 32), dtype=np.int32)
    spikes = np.ones((2, 32), dtype=bool)
    weights = np.zeros((3, 2), dtype=np.int8)
    out = np.empty_like(current)

    with pytest.raises(ValueError, match="bool lhs and int4 rhs"):
        _compile(_reversed_int4_matmul, (current, weights, spikes, out))

    with pytest.raises(ValueError, match="only valid in bool @ int4 contractions"):
        _compile(_int4_arithmetic, (weights, weights.copy()))


def test_rvne_quantized_dot_qemu_signed_weights_and_long_reduction(tmp_path):
    _require_sdk()
    rng = np.random.default_rng(17)
    rows, count = 7, 1025
    spikes = rng.integers(0, 2, (rows, count), dtype=np.int8).astype(bool)
    weights = rng.integers(-8, 8, (count, 1), dtype=np.int8)
    current = rng.integers(-100, 100, (rows, 1), dtype=np.int32)
    out = np.zeros_like(current)
    arrays = (current, spikes, weights, out)
    handle = materialize(
        _compile(_quantized_matmul, arrays), output_dir=tmp_path, mode="aot"
    )
    handle(*arrays)
    expected = current + (spikes.astype(np.int32) @ weights.astype(np.int32))
    np.testing.assert_array_equal(out, expected)


def test_rvne_quantized_dot_qemu_rejects_non_int4_weights(tmp_path):
    _require_sdk()
    current = np.zeros((42, 3), dtype=np.int32)
    spikes = np.ones((1, 32), dtype=bool)
    weights = np.full((32, 1), 8, dtype=np.int8)
    out = np.zeros_like(current)
    arrays = (current, spikes, weights, out)
    handle = materialize(
        _compile(_quantized_matmul, arrays), output_dir=tmp_path, mode="aot"
    )

    with pytest.raises(ValueError, match="values must be in \\[-8, 7\\]"):
        handle(*arrays)


def test_rvne_lif_qemu_preserves_explicit_state_across_timesteps(tmp_path):
    _require_sdk()
    count = 33
    current = np.arange(-16, 17, dtype=np.int32)
    voltage = np.arange(-32, 34, 2, dtype=np.int32)
    syn_current = np.arange(-48, 51, 3, dtype=np.int32)
    spike = np.zeros(count, dtype=np.int32)
    next_voltage = np.zeros_like(spike)
    next_current = np.zeros_like(spike)
    arrays = (current, voltage, syn_current, spike, next_voltage, next_current)
    handle = materialize(
        _compile(_lif_application, arrays), output_dir=tmp_path, mode="aot"
    )

    for _ in range(5):
        integrated = syn_current + current
        potential = voltage - (voltage >> 10) + (integrated >> 3)
        decayed = integrated - (integrated >> 4)
        fired = potential > 1
        handle(*arrays)
        np.testing.assert_array_equal(spike, fired.astype(np.int32))
        np.testing.assert_array_equal(next_voltage, np.where(fired, 0, potential))
        np.testing.assert_array_equal(next_current, np.where(fired, 0, decayed))
        voltage[:] = next_voltage
        syn_current[:] = next_current


def test_rvne_spiking_convolution_qemu_with_standard_matmul(tmp_path):
    _require_sdk()
    import torch
    import torch.nn.functional as functional

    rng = np.random.default_rng(31)
    inputs = rng.integers(0, 2, (1, 5, 6, 7), dtype=np.int32).astype(bool)
    weights = rng.integers(-8, 8, (3, 5, 3, 3), dtype=np.int8)
    padded_inputs = np.pad(inputs[0], ((0, 0), (1, 1), (1, 1)))
    patches = np.lib.stride_tricks.sliding_window_view(
        padded_inputs, (3, 3), axis=(1, 2)
    )
    patches = patches.transpose(1, 2, 0, 3, 4).reshape(42, 45)
    current = np.zeros((42, 3), dtype=np.int32)
    output = np.zeros_like(current)
    arrays = (current, patches, weights.reshape(3, 45).T.copy(), output)
    handle = materialize(
        _compile(_quantized_matmul, arrays), output_dir=tmp_path, mode="aot"
    )
    handle(*arrays)
    expected = (
        functional.conv2d(
            torch.from_numpy(inputs.astype(np.float32)),
            torch.from_numpy(weights.astype(np.float32)),
            padding=1,
        )
        .numpy()
        .astype(np.int32)
    )
    np.testing.assert_array_equal(output.T.reshape(1, 3, 6, 7), expected)
