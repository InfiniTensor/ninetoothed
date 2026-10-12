import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ninetoothed.backends.rvne_toolchain import find_rvne_toolchain

_CASES = (
    ((2, 1, 3, 4), (2, 1, 2, 2), (1, 1), (0, 0)),
    ((1, 2, 3, 5), (1, 2, 2, 3), (1, 2), (1, 1)),
)
_LEGACY_HEADER = """
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int conv_compute_currents_from_f32_ptr_defaultNP_f32out(
    int32_t T, int32_t CHI, int32_t HI, int32_t WI,
    int32_t CHO, int32_t K, int32_t STRIDE, int32_t PADDING,
    const float *input_linear, int64_t input_elems,
    const float *weight_linear, int64_t weight_elems,
    float *out_linear, int64_t out_elems);
int conv_compute_currents_from_f32_ptr_defaultNP_f32out_nonsquare(
    int32_t T, int32_t CHI, int32_t HI, int32_t WI, int32_t CHO,
    int32_t KH, int32_t KW, int32_t STRIDE_H, int32_t STRIDE_W,
    int32_t PAD_H, int32_t PAD_W,
    const float *input_linear, int64_t input_elems,
    const float *weight_linear, int64_t weight_elems,
    float *out_linear, int64_t out_elems);
#ifdef __cplusplus
}
#endif
"""


def _conv_data(case):
    input_shape, weight_shape, _, _ = _CASES[case]
    inputs = np.resize(
        np.array([-2.5, 0, 0.125, 3, -0.0], dtype=np.float32), input_shape
    )
    weights = np.resize(
        np.array(
            [-9.5, -7.5, -1.5, -0.5, 0.49, 0.5, 7.5, 9.5, -0.49, 1.5, 6.5, 0],
            dtype=np.float32,
        ),
        weight_shape,
    )

    return inputs, weights


def _conv_reference(case):
    inputs, weights = _conv_data(case)
    _, _, stride, padding = _CASES[case]
    spikes = (inputs != 0).astype(np.int32)
    quantized = np.clip(
        np.copysign(np.floor(np.abs(weights) + 0.5), weights), -8, 7
    ).astype(np.int32)
    padded = np.pad(
        spikes, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
    )
    batch, _, height, width = padded.shape
    channels, _, kernel_h, kernel_w = quantized.shape
    output = np.zeros(
        (
            batch,
            channels,
            (height - kernel_h) // stride[0] + 1,
            (width - kernel_w) // stride[1] + 1,
        ),
        dtype=np.float32,
    )

    for n, channel, row, col in np.ndindex(output.shape):
        patch = padded[
            n,
            :,
            row * stride[0] : row * stride[0] + kernel_h,
            col * stride[1] : col * stride[1] + kernel_w,
        ]
        output[n, channel, row, col] = np.sum(
            patch * quantized[channel], dtype=np.int64
        )

    return output


def _array_declaration(name, values):
    values = np.asarray(values)
    c_type = "float" if values.dtype == np.dtype("float32") else "int32_t"
    literals = (
        (f"{float(value)!r}f" if c_type == "float" else str(int(value)))
        for value in values.flat
    )

    return f"static const {c_type} {name}[] = {{{', '.join(literals)}}};"


@pytest.fixture(scope="module")
def rvne_sdk():
    if not os.environ.get("NINETOOTHED_RVNE_TOOLCHAIN"):
        pytest.skip("RVNE SDK is not configured")

    return find_rvne_toolchain()


@pytest.fixture(scope="module")
def native_bundle(tmp_path_factory, rvne_sdk):
    from examples.rvne.native import compile_native_bundle

    directory = tmp_path_factory.mktemp("rvne-native-abi")
    model_dir = os.environ.get("NINETOOTHED_SCONVLIF_MODEL_DIR")
    old_header = (
        None
        if model_dir is None
        else Path(model_dir) / "test_libs" / "sconv" / "conv_api.h"
    )
    header = (
        old_header.read_text(encoding="utf-8")
        if old_header is not None and old_header.is_file()
        else _LEGACY_HEADER
    )
    (directory / "legacy_conv_api.h").write_text(header, encoding="utf-8")
    declarations = []

    for index in range(2):
        inputs, weights = _conv_data(index)
        declarations.extend(
            (
                _array_declaration(f"inputs{index}", inputs),
                _array_declaration(f"weights{index}", weights),
            )
        )

    pool = np.arange(-140, 0, dtype=np.int32).reshape(2, 2, 5, 7)
    pool[:, :, -1, :] = 123
    pool[:, :, :, -1] = 456
    declarations.append(_array_declaration("pool_input", pool))
    driver = directory / "driver.c"
    driver.write_text(
        "#include <stdint.h>\n" + "\n".join(declarations) + "\n" + _DRIVER,
        encoding="utf-8",
    )
    config = {
        "convs": [
            {
                "input_shape": shape,
                "weights": np.zeros(weight_shape, dtype=np.int32),
                "stride": stride,
                "padding": padding,
            }
            for shape, weight_shape, stride, padding in _CASES
        ],
        "pools": [{"input_shape": pool.shape}],
        "lifs": [5],
    }
    bundle = compile_native_bundle(
        config, directory, extra_sources=(driver,), toolchain_root=str(rvne_sdk.root)
    )
    executable = Path(bundle["binary_path"])

    with executable.open("rb") as stream:
        elf = stream.read(20)

    assert elf[:6] == b"\x7fELF\x02\x01"
    assert int.from_bytes(elf[18:20], "little") == 243

    return bundle


def _run_native(bundle, toolchain, directory, mode):
    output = directory / "result.bin"
    subprocess.run(
        [
            str(toolchain.emulator),
            "-L",
            str(toolchain.sysroot),
            str(bundle["binary_path"]),
            mode,
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    return output


@pytest.mark.parametrize("case", (0, 1), ids=("square-batch2", "nonsquare"))
def test_rvne_standalone_old_conv_abi_qemu_quantizes_caller_weights(
    tmp_path, rvne_sdk, native_bundle, case
):
    output = _run_native(native_bundle, rvne_sdk, tmp_path, f"conv{case}")
    expected = _conv_reference(case)
    actual = np.fromfile(output, dtype=np.float32).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)


def test_rvne_standalone_old_conv_abi_qemu_rejects_invalid_calls(
    tmp_path, rvne_sdk, native_bundle
):
    output = _run_native(native_bundle, rvne_sdk, tmp_path, "invalid")
    data = output.read_bytes()
    statuses = np.frombuffer(data, dtype=np.int32, count=6)
    assert np.all(statuses < 0)
    np.testing.assert_array_equal(
        np.frombuffer(data, dtype=np.float32, offset=24),
        np.full(24, 123.0, dtype=np.float32),
    )


def test_rvne_standalone_conv_qemu_resets_prior_neuron_configuration(
    tmp_path, rvne_sdk, native_bundle
):
    output = _run_native(native_bundle, rvne_sdk, tmp_path, "contaminated")
    expected = _conv_reference(0)
    np.testing.assert_array_equal(
        np.fromfile(output, dtype=np.float32).reshape(expected.shape), expected
    )


def test_rvne_standalone_pool_and_lif_qemu_preserve_native_state(
    tmp_path, rvne_sdk, native_bundle
):
    output = _run_native(native_bundle, rvne_sdk, tmp_path, "pool-lif")
    actual = np.fromfile(output, dtype=np.int32)
    pool = np.arange(-140, 0, dtype=np.int32).reshape(2, 2, 5, 7)
    expected_pool = np.max(
        np.stack(
            [
                pool[:, :, row:4:2, col:6:2]
                for row, col in ((0, 0), (0, 1), (1, 0), (1, 1))
            ]
        ),
        axis=0,
    )
    expected = [expected_pool.reshape(-1)]
    voltage = np.array([-3, 0, 1, 2, 9], dtype=np.int32)
    syn_current = np.array([8, -4, 0, 16, -32], dtype=np.int32)

    for step in range(3):
        current = np.array([-5, 0, 8, 32, -17], dtype=np.int32) + step
        integrated = syn_current + current
        potential = voltage - (voltage >> 10) + (integrated >> 3)
        fired = potential > 1
        voltage = np.where(fired, 0, potential)
        syn_current = np.where(fired, 0, integrated - (integrated >> 4))
        expected.extend((fired.astype(np.int32), voltage, syn_current))

    np.testing.assert_array_equal(actual, np.concatenate(expected))


def test_rvne_standalone_full_model_qemu_uses_one_process(tmp_path, rvne_sdk):
    model_dir = os.environ.get("NINETOOTHED_SCONVLIF_MODEL_DIR")

    if not model_dir:
        pytest.skip("external SConvLif model is not configured")

    from examples.rvne.sconv_lif_native import verify_native_model

    report = verify_native_model(
        model_dir,
        tmp_path,
        toolchain_root=str(rvne_sdk.root),
        seed=20260830,
        verbose=False,
    )
    assert report["status"] == "passed"
    assert report["qemu_invocations"] == 1
    assert report["operator_calls"] == 88
    assert report["comparison_count"] == 165
    assert report["final_mismatches"] == 0
    assert report["final_nonzero"] == 108


def test_rvne_original_entry_qemu_links_without_source_changes(tmp_path, rvne_sdk):
    model_dir = os.environ.get("NINETOOTHED_SCONVLIF_MODEL_DIR")

    if not model_dir:
        pytest.skip("external SConvLif model is not configured")

    from examples.rvne import sconv_lif_native
    from examples.rvne.native import compile_native_bundle
    from examples.rvne.sconv_lif import _load_reference, _reference_run

    module, _ = _load_reference(model_dir)
    model, inputs, expected, _ = _reference_run(module, 20260830)
    entry = Path(model_dir).resolve() / "test_libs/main_test.c"
    source_hash = hashlib.sha256(entry.read_bytes()).hexdigest()
    driver = Path(sconv_lif_native.__file__).with_suffix(".cpp")
    bundle = compile_native_bundle(
        sconv_lif_native._model_config(model),
        tmp_path / "bundle",
        extra_sources=(driver, entry),
        toolchain_root=str(rvne_sdk.root),
        extra_flags=("-DNINETOOTHED_NO_MAIN",),
    )
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    executable = deployment / "model.elf"
    executable.write_bytes(Path(bundle["binary_path"]).read_bytes())
    executable.chmod(0o755)
    inputs.astype("<f4").tofile(deployment / "model_input.bin")
    expected.astype("<f4").tofile(deployment / "model_output.bin")
    result = subprocess.run(
        [
            str(rvne_sdk.emulator),
            "-L",
            str(rvne_sdk.sysroot),
            str(executable),
            "--strict",
        ],
        cwd=deployment,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert "mismatches=0/108" in result.stdout
    assert "matches Python reference exactly" in result.stdout
    assert hashlib.sha256(entry.read_bytes()).hexdigest() == source_hash
    assert bundle["source_hashes"][str(entry)] == source_hash
    assert {path.name for path in deployment.iterdir()} == {
        "model.elf",
        "model_input.bin",
        "model_output.bin",
    }


_DRIVER = r"""
#ifdef __cplusplus
#error The legacy client must be compiled as C.
#endif
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "native_runtime.h"
#include "legacy_conv_api.h"

static int square(const float *inputs, int64_t input_count,
                  const float *weights, int64_t weight_count,
                  float *out, int64_t output_count) {
  return conv_compute_currents_from_f32_ptr_defaultNP_f32out(
      2, 1, 3, 4, 2, 2, 1, 0, inputs, input_count, weights, weight_count, out, output_count);
}
static bool untouched(const float *output) {
  for (int i = 0; i < 36; ++i) if (output[i] != 123.0f) return false;
  return true;
}
int main(int argc, char **argv) {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  (void)cycles;
  if (argc != 3) return 60;
  FILE *file = fopen(argv[2], "wb");
  if (!file) return 61;
  if (!strcmp(argv[1], "invalid")) {
    float out[36];
    for (int i = 0; i < 36; ++i) out[i] = 123.0f;
    int32_t status[6];
    status[0] = square(inputs0, 23, weights0, 8, out, 24);
    if (!untouched(out)) return 70;
    status[1] = square(inputs0, 24, weights0, 7, out, 24);
    if (!untouched(out)) return 71;
    status[2] = square(inputs0, 24, weights0, 8, out, 23);
    if (!untouched(out)) return 72;
    float unknown_input[32] = {0}, unknown_weight[8] = {0};
    status[3] = conv_compute_currents_from_f32_ptr_defaultNP_f32out(
        2, 1, 4, 4, 2, 2, 1, 0, unknown_input, 32, unknown_weight, 8, out, 36);
    if (!untouched(out)) return 73;
    float bad_weights[8]; memcpy(bad_weights, weights0, sizeof(bad_weights));
    bad_weights[0] = NAN;
    status[4] = square(inputs0, 24, bad_weights, 8, out, 24);
    if (!untouched(out)) return 75;
    status[5] = square(inputs0, -1, weights0, 8, out, 24);
    if (!untouched(out)) return 76;
    if (fwrite(status, sizeof(int32_t), 6, file) != 6) return 62;
    if (fwrite(out, sizeof(float), 24, file) != 24) return 62;
  } else if (!strcmp(argv[1], "pool-lif")) {
    int32_t pooled[24];
    if (nt_pool_i32(&nt_pool_specs[0], pool_input, pooled)) return 80;
    if (fwrite(pooled, sizeof(int32_t), 24, file) != 24) return 62;
    int32_t voltage[5] = {-3, 0, 1, 2, 9};
    int32_t state[5] = {8, -4, 0, 16, -32};
    const int32_t baseline[5] = {-5, 0, 8, 32, -17};
    for (int t = 0; t < 3; ++t) {
      int32_t current[5], spike[5], next_voltage[5], next_current[5];
      for (int i = 0; i < 5; ++i) current[i] = baseline[i] + t;
      if (nt_lif_i32(&nt_lif_specs[0], current, voltage, state, spike, next_voltage, next_current)) return 81;
      if (fwrite(spike, sizeof(int32_t), 5, file) != 5) return 62;
      if (fwrite(next_voltage, sizeof(int32_t), 5, file) != 5) return 62;
      if (fwrite(next_current, sizeof(int32_t), 5, file) != 5) return 62;
      memcpy(voltage, next_voltage, sizeof(voltage));
      memcpy(state, next_current, sizeof(state));
    }
  } else {
    float out[24] = {0};
    int count = 24;
    int result;
    if (!strcmp(argv[1], "conv1")) {
      result = conv_compute_currents_from_f32_ptr_defaultNP_f32out_nonsquare(
          1, 2, 3, 5, 1, 2, 3, 1, 2, 1, 1, inputs1, 30, weights1, 12, out, 12);
      count = 12;
    } else {
      if (!strcmp(argv[1], "contaminated"))
        __builtin_riscv_set_lpr_64(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
      result = square(inputs0, 24, weights0, 8, out, 24);
    }
    if (result) return 82;
    if (fwrite(out, sizeof(float), count, file) != (size_t)count) return 62;
  }
  return fclose(file) == 0 ? 0 : 63;
}
"""
