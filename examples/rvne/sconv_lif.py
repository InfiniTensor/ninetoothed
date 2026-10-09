"""Verify the external SConvLif model with NineToothed kernels running in QEMU."""

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

from ninetoothed.backends.rvne_toolchain import rvne_compiler_identity

if __package__:
    from .operators import RvneOperators
else:
    from operators import RvneOperators


_STAGES = (
    ("layer1", "act1"),
    ("layer3", "act2"),
    ("layer5", "act3"),
    ("layer7", "act4"),
    ("layer9", "act5"),
    ("layer11", "act6"),
    ("layer12", "act7"),
    ("layer13", "act8"),
    ("layer14", "act9"),
)
_LIF_PARAMETERS = {
    "v_th": 1.0,
    "v_reset": 0.0,
    "tau_v": 1024.0,
    "tau_i": 16.0,
    "tau_vi": 8.0,
}


def _load_reference(model_dir):
    source = Path(model_dir).expanduser().resolve() / "yolo_origin.py"

    if not source.is_file():
        raise FileNotFoundError(f"SConvLif reference source is missing: {source}.")

    spec = importlib.util.spec_from_file_location("sconv_lif_reference", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module, source


def _reference_run(module, seed):
    import torch

    torch.set_num_threads(1)
    model = module.build_origin_model(seed)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_((parameter >= 0).to(parameter.dtype))

    _validate_model(model, module)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    inputs = torch.randint(0, 2, (4, 3, 64, 64), generator=generator).float()
    events = []
    hooks = []

    def capture(name):
        def callback(layer, args, output):
            del args
            event = {"name": name, "output": output.detach().numpy().copy()}

            if isinstance(layer, module.IntegerLIFNode):
                state = layer.v_combined.detach().numpy()
                neurons = output[0].numel()
                event["voltage"] = state[:, :neurons].reshape(output.shape).copy()
                event["current"] = state[:, neurons:].reshape(output.shape).copy()

            events.append(event)

        return callback

    for name, layer in model.named_modules():
        if (
            isinstance(
                layer, (torch.nn.Conv2d, torch.nn.MaxPool2d, module.IntegerLIFNode)
            )
            or name == "detect"
        ):
            hooks.append(layer.register_forward_hook(capture(name)))

    try:
        with torch.no_grad():
            output = model(inputs)
    finally:
        for hook in hooks:
            hook.remove()

    if len(events) != 92 or tuple(output.shape) != (1, 3, 2, 2, 9):
        raise ValueError("The reference does not match the supported SConvLif graph.")

    if not torch.count_nonzero(output).item():
        raise AssertionError("The reference fixture produced an all-zero output.")

    return model, inputs.numpy().astype(np.int32), output.numpy().copy(), events


def _validate_model(model, module):
    import torch

    dimensions = (module.T_STEPS, module.IN_CHANNELS, module.IN_HEIGHT, module.IN_WIDTH)

    if dimensions != (4, 3, 64, 64):
        raise ValueError("This adapter requires the four-step 3x64x64 SConvLif model.")

    convolutions = [model.get_submodule(name) for name, _ in _STAGES]
    convolutions.append(model.detect.conv)

    for layer in convolutions:
        if (
            not isinstance(layer, torch.nn.Conv2d)
            or layer.bias is not None
            or layer.groups != 1
            or layer.dilation != (1, 1)
        ):
            raise ValueError(
                "SConvLif convolutions must be bias-free with group and dilation one."
            )

    pool = model.pool

    if (
        not isinstance(pool, torch.nn.MaxPool2d)
        or pool.kernel_size != 2
        or pool.stride != 2
        or pool.padding != 0
        or pool.ceil_mode
    ):
        raise ValueError("SConvLif pooling must be 2x2 with stride two and no padding.")

    for _, name in _STAGES:
        layer = model.get_submodule(name)

        if not isinstance(layer, module.IntegerLIFNode) or any(
            getattr(layer, key) != value for key, value in _LIF_PARAMETERS.items()
        ):
            raise ValueError(f"Unsupported LIF parameters for `{name}`.")


def _compare(report, actual, expected, *, step, node, field):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    same_shape = actual.shape == expected.shape
    passed = same_shape and np.array_equal(actual, expected)
    mismatches = int(np.count_nonzero(actual != expected)) if same_shape else None
    max_error = None

    if same_shape and np.isfinite(actual).all() and np.isfinite(expected).all():
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        max_error = float(np.max(np.abs(difference), initial=0))

    report["comparisons"].append(
        {
            "step": step,
            "node": node,
            "field": field,
            "shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "passed": passed,
            "mismatches": mismatches,
            "max_abs_error": max_error,
        }
    )

    if not passed:
        raise AssertionError(
            f"Mismatch at step {step}, {node}.{field}: {mismatches} values, max error {max_error}."
        )


def _write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_model(
    model_dir, output_dir, *, toolchain_root=None, seed=20260830, verbose=True
):
    """Run all model math in QEMU and compare every layer and neuron state."""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    report = {
        "status": "running",
        "model": "SConvLif",
        "seed": seed,
        "weight_policy": "Deterministic binary integration fixture; no trained checkpoint.",
        "execution": "Python scheduling and packing; NineToothed AOT kernels through QEMU.",
        "comparisons": [],
    }
    _write_report(report_path, report)
    operators = None
    started = time.perf_counter()

    try:
        report["toolchain"] = rvne_compiler_identity(toolchain_root, required=True)
        module, source = _load_reference(model_dir)
        report["reference_source"] = str(source)
        report["reference_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        model, inputs, expected, events = _reference_run(module, seed)
        report["input_shape"] = list(inputs.shape)
        report["output_shape"] = list(expected.shape)
        report["parameter_count"] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        report["timesteps"] = 4
        np.savez(directory / "reference.npz", inputs=inputs, output=expected)
        operators = RvneOperators(directory / "kernels", toolchain_root=toolchain_root)
        layer_names = [name for name, _ in _STAGES] + ["detect.conv"]
        weights = {
            name: operators.prepare_weights(
                model.get_submodule(name).weight.detach().numpy().astype(np.int32)
            )
            for name in layer_names
        }
        states = {}
        event_index = 0

        def check(name, value, step, state=None):
            nonlocal event_index
            event = events[event_index]

            if event["name"] != name:
                raise AssertionError(
                    f"Model execution order differs: expected {event['name']}, got {name}."
                )

            _compare(
                report, value, event["output"], step=step, node=name, field="output"
            )

            if state is not None:
                _compare(
                    report,
                    state[0],
                    event["voltage"],
                    step=step,
                    node=name,
                    field="voltage",
                )
                _compare(
                    report,
                    state[1],
                    event["current"],
                    step=step,
                    node=name,
                    field="current",
                )

            event_index += 1

            if verbose:
                print(
                    f"step {step + 1}/4  {name:12s}  {str(tuple(value.shape)):20s}  exact",
                    flush=True,
                )

        for step in range(4):
            x = inputs[step : step + 1].copy()

            for index, (conv_name, lif_name) in enumerate(_STAGES):
                convolution = model.get_submodule(conv_name)
                x = operators.conv2d(
                    x,
                    weights[conv_name],
                    stride=convolution.stride,
                    padding=convolution.padding,
                )
                check(conv_name, x, step)

                if index < 3:
                    x = operators.max_pool2d(x)
                    check("pool", x, step)

                if lif_name not in states:
                    states[lif_name] = (np.zeros_like(x), np.zeros_like(x))

                voltage, syn_current = states[lif_name]
                x, next_voltage, next_current = operators.lif(x, voltage, syn_current)
                states[lif_name] = next_voltage, next_current
                check(lif_name, x, step, states[lif_name])

            convolution = model.detect.conv
            x = operators.conv2d(
                x,
                weights["detect.conv"],
                stride=convolution.stride,
                padding=convolution.padding,
            )
            check("detect.conv", x, step)
            batch, _, height, width = x.shape
            actual = x.reshape(
                batch, model.detect.num_anchors, model.detect.no, height, width
            )
            actual = actual.transpose(0, 1, 3, 4, 2).copy()
            check("detect", actual, step)

        _compare(report, actual, expected, step=3, node="model", field="final_output")

        if event_index != len(events) or operators.launch_count != 88:
            raise AssertionError(
                "The full model did not execute all 88 QEMU operators."
            )

        if operators.launch_counts != {"conv2d": 40, "max_pool2d": 12, "lif": 36}:
            raise AssertionError("Unexpected SConvLif operator counts.")

        state_outputs = {
            f"{name}_{field}": array
            for name, arrays in states.items()
            for field, array in zip(("voltage", "current"), arrays)
        }
        np.savez(directory / "qemu_output.npz", output=actual, **state_outputs)
        report["final_mismatches"] = 0
        report["final_nonzero"] = int(np.count_nonzero(actual))
        report["final_range"] = [int(actual.min()), int(actual.max())]
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["comparison_count"] = len(report["comparisons"])

        if operators is not None:
            report["launch_count"] = operators.launch_count
            report["launch_counts"] = operators.launch_counts
            report["artifacts"] = operators.artifacts
            report["max_staging_bytes"] = operators.max_staging_bytes
            report["staging_records"] = operators.staging_records

        _write_report(report_path, report)

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", default=os.environ.get("NINETOOTHED_SCONVLIF_MODEL_DIR")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("build/rvne-sconv-lif"))
    parser.add_argument("--toolchain-root")
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    if not args.model_dir:
        parser.error("Pass --model-dir or set NINETOOTHED_SCONVLIF_MODEL_DIR.")

    report = verify_model(
        args.model_dir,
        args.output_dir,
        toolchain_root=args.toolchain_root,
        seed=args.seed,
    )
    print(
        f"PASS: {report['launch_count']} QEMU launches, {report['comparison_count']} exact comparisons, {report['final_nonzero']} nonzero output values."
    )
    print(f"Report: {(args.output_dir / 'report.json').resolve()}")


if __name__ == "__main__":
    main()
