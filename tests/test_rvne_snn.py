import os

import numpy as np
import pytest

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import CompileRequest, compile_kernel
from ninetoothed.compiler.runtime import materialize
from ninetoothed.rvne import pack_spikes, pack_weights


def _identity(*tensors):
    return tensors


def _spike_application(current, spikes, low, high, out):
    out = ntl.spike_accumulate(current, spikes, low, high)  # noqa: F841


def _dot_application(current, spikes, low, high, out):
    acc = current

    for block in range(spikes.shape[0]):
        acc = ntl.spike_accumulate(acc, spikes[block], low[block], high[block])

    out = acc  # noqa: F841


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
            tensors=tuple(Tensor(shape=a.shape, dtype=a.dtype.name) for a in arrays),
            backend="rvne",
            caller="numpy",
        )
    )


def _require_sdk():
    if not os.environ.get("NINETOOTHED_RVNE_TOOLCHAIN"):
        pytest.skip("RVNE SDK is not configured")


@pytest.mark.parametrize("count", (0, 1, 15, 16, 31, 32, 33, 1023, 1024, 1025))
def test_packing_preserves_bit_order_signed_weights_and_zero_tail(count):
    spikes = np.arange(count) % 2
    weights = (np.arange(count) % 16 - 8).astype(np.int32)
    packed_spikes = pack_spikes(spikes)
    packed_weights = pack_weights(weights)
    assert packed_spikes.dtype == np.uint32
    assert packed_weights.dtype == np.uint64

    for index in range(packed_spikes.size * 32):
        actual = (int(packed_spikes[index // 32]) >> (index % 32)) & 1
        assert actual == (int(spikes[index]) if index < count else 0)

    for index in range(packed_weights.size * 16):
        actual = (int(packed_weights[index // 16]) >> (4 * (index % 16))) & 15
        actual = actual - 16 if actual >= 8 else actual
        assert actual == (int(weights[index]) if index < count else 0)


@pytest.mark.parametrize(
    "function,values,error",
    (
        (pack_spikes, [0, 2], ValueError),
        (pack_spikes, [-1, 0], ValueError),
        (pack_weights, [-9, 7], ValueError),
        (pack_weights, [-8, 8], ValueError),
        (pack_weights, [0.5], TypeError),
        (pack_spikes, [0.0, 1.0], TypeError),
        (pack_weights, 1, ValueError),
    ),
)
def test_packing_rejects_implicit_quantization(function, values, error):
    with pytest.raises(error):
        function(values)


def test_spike_primitive_rejects_unpacked_weights():
    arrays = (
        np.zeros(3, dtype=np.int32),
        np.zeros(3, dtype=np.uint32),
        np.zeros(3, dtype=np.int32),
        np.zeros(3, dtype=np.uint64),
        np.zeros(3, dtype=np.int32),
    )

    with pytest.raises(TypeError, match="packed weight"):
        _compile(_spike_application, arrays)


def test_rvne_spike_intrinsic_qemu_signed_weights_and_accumulation(tmp_path):
    _require_sdk()
    rng = np.random.default_rng(17)
    spikes = rng.integers(0, 2, (1025, 32), dtype=np.int32)
    weights = rng.integers(-8, 8, (1025, 32), dtype=np.int32)
    current = rng.integers(-100, 100, 1025, dtype=np.int32)
    spikes[:2] = 1
    weights[0] = 7
    weights[1] = -8
    current[:2] = (np.iinfo(np.int32).max, np.iinfo(np.int32).min)
    packed = pack_weights(weights)
    out = np.zeros_like(current)
    arrays = (
        current,
        pack_spikes(spikes)[:, 0].copy(),
        packed[:, 0].copy(),
        packed[:, 1].copy(),
        out,
    )
    compilation = _compile(_spike_application, arrays)
    assert "__builtin_riscv_calc_acc_32" in compilation.artifact.primary_source
    handle = materialize(compilation, output_dir=tmp_path, mode="aot")
    handle(*arrays)
    expected = current + np.sum(spikes * weights, axis=1, dtype=np.int32)
    np.testing.assert_array_equal(out, expected)


def test_rvne_packed_dot_qemu_exceeds_register_capacity(tmp_path):
    _require_sdk()
    rng = np.random.default_rng(21)
    count, rows = 1025, 7
    spikes = rng.integers(0, 2, (rows, count), dtype=np.int32)
    weights = rng.integers(-8, 8, (rows, count), dtype=np.int32)
    padding = (-count) % 32
    packed = pack_weights(np.pad(weights, ((0, 0), (0, padding))))
    current = np.arange(rows, dtype=np.int32)
    out = np.zeros_like(current)
    arrays = (
        current,
        pack_spikes(spikes).T.copy(),
        packed[:, ::2].T.copy(),
        packed[:, 1::2].T.copy(),
        out,
    )
    handle = materialize(
        _compile(_dot_application, arrays), output_dir=tmp_path, mode="aot"
    )
    handle(*arrays)
    expected = current + np.sum(spikes * weights, axis=1, dtype=np.int32)
    np.testing.assert_array_equal(out, expected)


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


def test_rvne_spiking_convolution_qemu_with_explicit_packed_patches(tmp_path):
    _require_sdk()
    import torch
    import torch.nn.functional as functional

    rng = np.random.default_rng(31)
    inputs = rng.integers(0, 2, (1, 5, 6, 7), dtype=np.int32)
    weights = rng.integers(-8, 8, (3, 5, 3, 3), dtype=np.int32)
    padded_inputs = np.pad(inputs[0], ((0, 0), (1, 1), (1, 1)))
    patches = np.lib.stride_tricks.sliding_window_view(
        padded_inputs, (3, 3), axis=(1, 2)
    )
    patches = patches.transpose(1, 2, 0, 3, 4).reshape(42, 45)
    spikes = np.tile(patches, (3, 1))
    row_weights = np.repeat(weights.reshape(3, 45), 42, axis=0)
    packed = pack_weights(np.pad(row_weights, ((0, 0), (0, 19))))
    current = np.zeros(126, dtype=np.int32)
    output = np.zeros_like(current)
    arrays = (
        current,
        pack_spikes(spikes).T.copy(),
        packed[:, ::2].T.copy(),
        packed[:, 1::2].T.copy(),
        output,
    )
    handle = materialize(
        _compile(_dot_application, arrays), output_dir=tmp_path, mode="aot"
    )
    handle(*arrays)
    expected = (
        functional.conv2d(
            torch.from_numpy(inputs).float(),
            torch.from_numpy(weights).float(),
            padding=1,
        )
        .numpy()
        .astype(np.int32)
    )
    np.testing.assert_array_equal(output.reshape(1, 3, 6, 7), expected)
