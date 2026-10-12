"""Build and verify SConvLif as one RISC-V ELF with no Python inference runtime."""

import argparse
import hashlib
import os
import re
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

from examples.rvne.native import compile_native_bundle
from examples.rvne.sconv_lif import (
    _STAGES,
    _compare,
    _load_reference,
    _reference_run,
    _write_report,
)
from ninetoothed.backends.rvne_toolchain import find_rvne_toolchain


def _model_config(model):
    config = {"convs": [], "pools": [], "lifs": []}
    shape = (1, 3, 64, 64)

    for index, name in enumerate([name for name, _ in _STAGES] + ["detect.conv"]):
        layer = model.get_submodule(name)
        weights = layer.weight.detach().numpy().astype(np.int32)
        config["convs"].append(
            {
                "input_shape": shape,
                "weights": weights,
                "stride": layer.stride,
                "padding": layer.padding,
            }
        )
        height = (
            shape[2] + 2 * layer.padding[0] - layer.kernel_size[0]
        ) // layer.stride[0] + 1
        width = (
            shape[3] + 2 * layer.padding[1] - layer.kernel_size[1]
        ) // layer.stride[1] + 1
        shape = (1, layer.out_channels, height, width)

        if index < 3:
            config["pools"].append({"input_shape": shape})
            shape = (1, shape[1], shape[2] // 2, shape[3] // 2)

        if index < 9:
            config["lifs"].append(int(np.prod(shape)))

    return config


def _compare_trace(path, events, report):
    record_index = 0
    step = -1

    with path.open("rb") as file:
        for event in events:
            if event["name"] == "layer1":
                step += 1

            for field in ("output", "voltage", "current"):
                if field not in event:
                    continue

                expected = event[field]
                header = file.read(8)

                if len(header) != 8:
                    raise ValueError(f"Missing native trace record {record_index}.")

                index, count = struct.unpack("<II", header)

                if index != record_index or count != expected.size:
                    raise ValueError(
                        f"Native trace record {record_index} has an invalid index or size."
                    )

                data = file.read(count * 4)

                if len(data) != count * 4:
                    raise ValueError(
                        f"Native trace record {record_index} is truncated."
                    )

                actual = np.frombuffer(data, dtype="<i4").reshape(expected.shape)
                _compare(
                    report, actual, expected, step=step, node=event["name"], field=field
                )
                record_index += 1

        if file.read(1):
            raise ValueError("Unexpected extra native trace records.")

    return record_index


def verify_native_model(
    model_dir,
    output_dir,
    *,
    toolchain_root=None,
    seed=20260830,
    verbose=True,
    build_only=False,
):
    """Build one ELF and compare one QEMU invocation against every reference stage."""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    report = {
        "status": "running",
        "model": "SConvLif",
        "execution": "One RV64 ELF; native C++ graph scheduling, state and operator calls.",
        "graph_origin": "SConvLif adapter, not regenerated original ONNX/MLIR graph code.",
        "seed": seed,
        "weight_policy": "Deterministic binary integration fixture; no trained checkpoint.",
        "qemu_invocations": 0,
        "comparisons": [],
    }
    _write_report(report_path, report)
    started = time.perf_counter()

    try:
        toolchain = find_rvne_toolchain(toolchain_root)
        module, source = _load_reference(model_dir)
        report["reference_source"] = str(source)
        report["reference_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        model, inputs, expected, events = _reference_run(module, seed)
        report["input_shape"] = list(inputs.shape)
        report["output_shape"] = list(expected.shape)
        input_path = directory / "input.bin"
        output_path = directory / "output.bin"
        trace_path = directory / "trace.bin"
        inputs.astype("<f4").tofile(input_path)
        expected.astype("<f4").tofile(directory / "expected.bin")
        np.savez(directory / "reference.npz", inputs=inputs, output=expected)
        driver = Path(__file__).with_suffix(".cpp")
        bundle = compile_native_bundle(
            _model_config(model),
            directory / "bundle",
            extra_sources=(driver,),
            toolchain_root=str(toolchain.root),
            binary_name="sconv_lif.elf",
        )
        report["bundle"] = bundle
        command = [
            str(toolchain.emulator),
            "-L",
            str(toolchain.sysroot),
            bundle["binary_path"],
            str(input_path),
            str(output_path),
            str(trace_path),
        ]
        report["run_command"] = command

        if build_only:
            report["status"] = "built"
        else:
            if verbose:
                print(
                    "Running the complete four-step graph in one QEMU process...",
                    flush=True,
                )

            inference_started = time.perf_counter()
            report["qemu_invocations"] += 1
            result = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=300
            )
            report["execution_seconds"] = time.perf_counter() - inference_started
            report["program_stdout"] = result.stdout
            counts = re.search(
                r"operator_calls=(\d+) conv2d=(\d+) max_pool2d=(\d+) lif=(\d+) trace_records=(\d+)",
                result.stdout,
            )

            if counts is None or tuple(map(int, counts.groups())) != (
                88,
                40,
                12,
                36,
                164,
            ):
                raise AssertionError(
                    "Native executable did not report the complete model operation counts."
                )

            report["operator_calls"] = 88
            report["operator_counts"] = {"conv2d": 40, "max_pool2d": 12, "lif": 36}

            if output_path.stat().st_size != expected.size * 4:
                raise ValueError("Native output file has an unexpected size.")

            actual = np.fromfile(output_path, dtype="<f4").reshape(expected.shape)
            report["trace_records"] = _compare_trace(trace_path, events, report)
            _compare(
                report, actual, expected, step=3, node="model", field="final_output"
            )
            np.savez(directory / "qemu_output.npz", output=actual)
            report["final_mismatches"] = 0
            report["final_nonzero"] = int(np.count_nonzero(actual))
            report["final_range"] = [float(actual.min()), float(actual.max())]
            report["status"] = "passed"

            if verbose:
                print(result.stdout.strip(), flush=True)
                print(
                    "PASS: one QEMU invocation, 88 native operator calls, 165 exact comparisons.",
                    flush=True,
                )
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"

        if isinstance(error, subprocess.CalledProcessError):
            report["program_stderr"] = error.stderr

        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["comparison_count"] = len(report["comparisons"])
        _write_report(report_path, report)

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", default=os.environ.get("NINETOOTHED_SCONVLIF_MODEL_DIR")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("build/rvne-sconv-lif-native")
    )
    parser.add_argument("--toolchain-root")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    if not args.model_dir:
        parser.error("Pass --model-dir or set NINETOOTHED_SCONVLIF_MODEL_DIR.")

    report = verify_native_model(
        args.model_dir,
        args.output_dir,
        toolchain_root=args.toolchain_root,
        seed=args.seed,
        build_only=args.build_only,
    )
    print(f"Executable: {report['bundle']['binary_path']}")
    print(f"Report: {(args.output_dir / 'report.json').resolve()}")


if __name__ == "__main__":
    main()
