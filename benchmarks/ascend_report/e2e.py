"""Run resumable, ordered generation measurements and derive report data."""

import argparse
import collections
import contextlib
import datetime
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if __package__:
    from .affinity import plan_cpu_affinity
    from .e2e_runner import (
        ORDERS,
        SCOPES,
        read_config,
        reference_path,
        sha256_file,
        write_json,
    )
else:
    from affinity import plan_cpu_affinity
    from e2e_runner import (
        ORDERS,
        SCOPES,
        read_config,
        reference_path,
        sha256_file,
        write_json,
    )


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def git_revision(directory):
    try:
        return subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_identity(config):
    """Hash the actual compiler, adapters, baseline and orchestration sources."""
    roots = {
        "ninetoothed": Path(config["paths"]["ninetoothed"]) / "src" / "ninetoothed",
        "ntops": Path(config["paths"]["ntops"]) / "src" / "ntops_lab",
        "baseline": Path(config["paths"]["baseline"]) / "triton_ascend_ops",
    }
    files = {}

    for name, root in roots.items():
        if not root.is_dir():
            raise ValueError(f"Source directory does not exist: {root}")

        for path in sorted(root.rglob("*.py")):
            files[name + "/" + path.relative_to(root).as_posix()] = sha256_file(path)

    for path in sorted(Path(__file__).parent.glob("*.py")):
        files["runner/" + path.name] = sha256_file(path)

    return {
        "sha256": files,
        "commits": {
            name: git_revision(config["paths"][name])
            for name in ("ninetoothed", "ntops", "baseline")
        },
    }


def model_identity(config):
    """Hash local model assets once before any timed worker starts."""
    root = Path(config["paths"]["model"])

    if not (root / "config.json").is_file():
        raise ValueError(f"Model config is missing: {root / 'config.json'}")

    files = {}
    suffixes = {".json", ".safetensors", ".bin", ".model", ".txt", ".jinja"}

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)

        if (
            path.is_file()
            and path.suffix in suffixes
            and not any(part.startswith(".") for part in relative.parts)
        ):
            files[relative.as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    if not any(Path(name).suffix in {".bin", ".safetensors"} for name in files):
        raise ValueError("No local model weights were found.")

    return {
        "id": config["e2e"]["model_id"],
        "revision": config["e2e"]["model_revision"],
        "files": files,
    }


def identity_digest(config, sources, model, execution=None):
    payload = json.dumps(
        {"config": config, "sources": sources, "model": model, "execution": execution},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def record_path(root, order, stage, backend, device=0):
    if backend == "reference":
        return root / "references" / f"device-{device}.json"

    return root / "results" / f"{order}_stage{stage}_{backend}.json"


def validate_record(record, identity, settings, device=None, cpu_affinity=None):
    if record.get("run_identity") != identity:
        raise ValueError("Existing worker result has a different run identity.")

    if record.get("status") != "PASS":
        return False

    if record.get("worker_returncode") != 0:
        return False

    if device is not None:
        runtime = record.get("environment", {})

        if runtime.get("physical_device") != device or not runtime.get("device_uuid"):
            raise ValueError("Worker physical device or UUID does not match its lane.")

        if cpu_affinity is not None and runtime.get("cpu_affinity") != cpu_affinity:
            raise ValueError("Worker CPU affinity does not match its lane.")

    if not record.get("sanity", {}).get("passed"):
        raise ValueError("A PASS result is missing successful text sanity checks.")

    if record["model"]["input_tokens"] != settings["input_tokens"]:
        raise ValueError("Existing worker input length differs from the configuration.")

    if len(record.get("output_ids", [])) != settings["output_tokens"]:
        raise ValueError(
            "Existing worker output length differs from the configuration."
        )

    if record.get("backend") == "reference":
        return True

    samples = record.get("samples_seconds", [])

    if len(samples) != settings["samples"] or not all(
        isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        for value in samples
    ):
        raise ValueError("Existing timing samples are incomplete or invalid.")

    if statistics.median(samples) != record.get("median_seconds"):
        raise ValueError("Existing median does not match its raw samples.")

    if len(record.get("sample_outputs", [])) != len(samples):
        raise ValueError("Existing samples are missing their output diagnostics.")

    return True


def run_process(command, log, timeout):
    """Stop a timed-out worker and its compiler children before starting another."""
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )

    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

        return 124


def logits_diagnostics(first, second):
    """Compare optional compact logits as diagnostics, never as an acceptance gate."""
    import numpy as np

    first_path = Path(first["logits_diagnostic"]["path"])
    second_path = Path(second["logits_diagnostic"]["path"])
    results = {}

    for record, path in ((first, first_path), (second, second_path)):
        if sha256_file(path) != record["logits_diagnostic"]["sha256"]:
            raise ValueError(f"Logits artifact changed: {path}")

    with (
        np.load(first_path, allow_pickle=False) as left,
        np.load(second_path, allow_pickle=False) as right,
    ):
        for key in ("prefill", "decode"):
            a, b = left[key], right[key]

            if a.shape != b.shape:
                results[key] = {"status": "SHAPE_MISMATCH", "passed": False}

                continue

            matches = np.isclose(b, a, rtol=0.02, atol=0.1, equal_nan=False)
            finite = np.isfinite(a) & np.isfinite(b)
            errors = np.abs(a[finite] - b[finite])
            results[key] = {
                "passed": bool(matches.all()),
                "mismatch_count": int((~matches).sum()),
                "max_abs_error": float(errors.max()) if errors.size else None,
                "rtol": 0.02,
                "atol": 0.1,
                "tensor_shape": list(a.shape),
            }

    return {
        "comparison_scope": "last_position_logits",
        "acceptance_gate": False,
        **results,
    }


def semantic_review(root):
    path = root / "semantic-review.json"

    if not path.exists():
        return {}

    value = json.loads(path.read_text(encoding="utf-8"))

    return value.get("records", {})


def acceptance_state(root, records, review):
    """Require an explicit, hash-bound human review for normal coherent text."""
    states = []

    for path, record in records:
        key = path.relative_to(root).as_posix()
        item = review.get(key)

        if not record.get("sanity", {}).get("passed"):
            states.append(False)
        elif item is None:
            states.append(None)
        elif item.get("raw_sha256") != sha256_file(path):
            raise ValueError(f"Semantic review belongs to an older raw result: {key}")
        elif not isinstance(item.get("passed"), bool):
            raise ValueError(
                f"Semantic review must contain a Boolean passed value: {key}"
            )
        else:
            states.append(item["passed"])

    if False in states:
        return False

    return None if None in states else True


def build_report(config, environment):
    root = Path(config["e2e_output_dir"])
    execution = environment["execution"]
    selected_orders = config["e2e"]["orders"]
    references = {}

    for device in execution["devices"]:
        path = reference_path(config, device)
        expected_hash = environment.get("result_sha256", {}).get(
            path.relative_to(root).as_posix()
        )

        if not path.exists() or expected_hash is None:
            continue

        if sha256_file(path) != expected_hash:
            raise ValueError("The recorded reference artifact changed.")

        reference = json.loads(path.read_text(encoding="utf-8"))

        if reference.get("status") == "PASS":
            validate_record(
                reference,
                environment["run_identity"],
                config["e2e"],
                device,
                execution["cpu_affinity"][str(device)],
            )
            references[str(device)] = reference

    if not references:
        return None

    reference = next(iter(references.values()))

    review = semantic_review(root)
    stages = []

    for stage in config["e2e"]["stages"]:
        device = execution["stage_devices"][str(stage)]
        stage_reference = references.get(str(device))
        timings = {}
        counts = {}
        native_matrix_counts = {}
        core_adapters = {}
        core_coverage = []
        diagnostics = {}
        output_texts = {}
        token_matches = []
        text_matches = []
        acceptance = []
        sanity = []

        for label in selected_orders:
            order, _ = ORDERS[label]
            paths = {
                backend: record_path(root, order, stage, backend)
                for backend in ("triton", "ntops")
            }

            if stage_reference is None or not all(
                path.exists()
                and path.relative_to(root).as_posix() in environment["result_sha256"]
                for path in paths.values()
            ):
                timings[label] = None

                continue

            records = {
                backend: json.loads(path.read_text(encoding="utf-8"))
                for backend, path in paths.items()
            }

            for path in paths.values():
                expected_hash = environment.get("result_sha256", {}).get(
                    path.relative_to(root).as_posix()
                )

                if expected_hash and sha256_file(path) != expected_hash:
                    raise ValueError(f"The recorded raw artifact changed: {path}")

            if not all(record.get("status") == "PASS" for record in records.values()):
                timings[label] = {
                    "status": "ERROR",
                    "worker_statuses": {
                        key: value.get("status") for key, value in records.items()
                    },
                    "errors": {
                        key: value.get("error") for key, value in records.items()
                    },
                }

                continue

            for record in records.values():
                validate_record(
                    record,
                    environment["run_identity"],
                    config["e2e"],
                    device,
                    execution["cpu_affinity"][str(device)],
                )

                if (
                    record["environment"]["device_uuid"]
                    != stage_reference["environment"]["device_uuid"]
                ):
                    raise ValueError(
                        "Paired workers and their reference used different physical UUIDs."
                    )

            left, right = records["triton"], records["ntops"]
            core_coverage.extend(
                record.get("core_kernel_coverage", {}) for record in (left, right)
            )
            eligible = acceptance_state(
                root, [(paths[key], records[key]) for key in paths], review
            )
            token_matches.append(
                all(
                    output["output_ids"] == left["output_ids"]
                    for record in (left, right)
                    for output in record["sample_outputs"]
                )
            )
            text_matches.append(
                all(
                    output["output_text"] == left["output_text"]
                    for record in (left, right)
                    for output in record["sample_outputs"]
                )
            )
            acceptance.append(eligible)
            sanity.append(
                all(record["sanity"]["passed"] for record in records.values())
            )
            timings[label] = {
                "triton_median_s": left["median_seconds"],
                "ninetoothed_median_s": right["median_seconds"],
                "ratio_percent": 100 * left["median_seconds"] / right["median_seconds"],
                "measurement_role": "qualified"
                if eligible
                else "text_output_review_pending"
                if eligible is None
                else "diagnostic_text_failed",
                "acceptance_eligible": eligible,
                "physical_device": device,
                "device_uuid": stage_reference["environment"]["device_uuid"],
                "cpu_affinity": stage_reference["environment"]["cpu_affinity"],
                "tokens_equal": token_matches[-1],
                "text_equal": text_matches[-1],
                "operator_scopes_equal": left["operator_calls"]
                == right["operator_calls"],
                "raw_files": {
                    key: str(path.relative_to(root)) for key, path in paths.items()
                },
            }
            counts[label] = {
                "triton": left["operator_calls"],
                "ninetoothed": right["operator_calls"],
            }
            native_matrix_counts[label] = {
                "triton": left.get("native_matrix_calls", {}),
                "ninetoothed": right.get("native_matrix_calls", {}),
            }
            core_adapters[label] = {
                "triton": left.get("core_operator_adapters", {}),
                "ninetoothed": right.get("core_operator_adapters", {}),
            }
            output_texts[label] = {
                "triton": left["output_text"],
                "ninetoothed": right["output_text"],
            }

            if config["e2e"]["logits_diagnostics"]:
                diagnostics[label] = logits_diagnostics(left, right)

        stages.append(
            {
                "stage": stage,
                "scope": SCOPES[stage],
                "selected_orders": selected_orders,
                "physical_device": device,
                "device_uuid": stage_reference["environment"]["device_uuid"]
                if stage_reference
                else None,
                "accuracy": {
                    "comparison_scope": "normal_coherent_text",
                    "sanity_passed": all(sanity) if sanity else None,
                    "tokens_equal": all(token_matches) if token_matches else None,
                    "text_equal": all(text_matches) if text_matches else None,
                    "passed": False
                    if False in acceptance
                    else True
                    if len(acceptance) == len(selected_orders)
                    and all(value is True for value in acceptance)
                    else None,
                    "token_and_text_equality_are_acceptance_gates": False,
                    "token_and_text_comparison": "All sampled outputs within each backend pair.",
                    "logits_are_acceptance_gate": False,
                    "semantic_review": "semantic-review.json",
                },
                "operator_counts": counts.get("forward", counts.get("reverse", {})),
                "operator_counts_by_order": counts,
                "native_matrix_counts_by_order": native_matrix_counts,
                "core_operator_adapters": core_adapters,
                "replacement_coverage": {
                    "sdpa_requested": stage >= 5,
                    "sdpa_verified": stage >= 5
                    and len(core_coverage) == 2 * len(selected_orders)
                    and all(
                        item.get("sdpa", {}).get("verified") is True
                        for item in core_coverage
                    ),
                    "model_linear_projections_and_lm_head_requested": stage >= 6,
                    "model_linear_projections_and_lm_head_verified": stage >= 6
                    and len(core_coverage) == 2 * len(selected_orders)
                    and all(
                        item.get("linear_mm", {}).get("verified") is True
                        for item in core_coverage
                    ),
                    "all_model_operators_replaced": False,
                },
                "output_texts": output_texts,
                "logits_diagnostics": diagnostics,
                "timings": timings,
            }
        )

    report = {
        "selected_orders": selected_orders,
        "model": reference["model"],
        "reference_by_device": {
            device: {
                "raw_file": reference_path(config, int(device))
                .relative_to(root)
                .as_posix(),
                "physical_device": int(device),
                "device_uuid": record["environment"]["device_uuid"],
                "cpu_affinity": record["environment"]["cpu_affinity"],
                "output_text": record["output_text"],
            }
            for device, record in references.items()
        },
        "stages": stages,
        "provenance": {
            "run_identity": environment["run_identity"],
            "source_identity": environment["sources"],
            "model_identity": environment["model"],
            "reference_environment": reference["environment"],
            "reference_environments_by_device": {
                key: value["environment"] for key, value in references.items()
            },
            "execution": execution,
            "observed_max_concurrent_workers": environment.get(
                "observed_max_concurrent_workers", 0
            ),
            "sampling": {
                "orders": selected_orders,
                "warmup_per_process": config["e2e"]["warmup"],
                "samples_per_process": config["e2e"]["samples"],
                "statistic": "median",
                "timing": "Synchronize, start timer, complete generate, synchronize, stop timer.",
                "excluded": [
                    "model load",
                    "tokenization",
                    "one-time weight preparation and restoration",
                    "output copy/check/text decoding",
                    "launch counting",
                ],
                "independent_process_per": "order/stage/backend",
                "TRITON_ALL_BLOCKS_PARALLEL": "0",
                "cpu_threads": config["e2e"]["cpu_threads"],
                "linear_tuning": config["e2e"]["linear_tuning"],
                "linear_block_sizes": config["e2e"]["linear_block_sizes"],
                "linear_weight_layout": config["e2e"]["linear_weight_layout"],
            },
            "ratio_percent_formula": "100 * triton_median_s / ninetoothed_median_s; higher is better",
            "semantic_acceptance": "Human review of normal coherent output; logits and exact token equality are diagnostic only.",
            "preserved": [
                "HF Llama model orchestration",
                "HF causal mask",
                "HF KV cache",
                "Embedding, generation control and unpatched primitive operations",
            ],
            "ntops_compatibility": [
                "launcher alias",
                "Tensor.to lowering",
                "shape adapters",
            ],
            "sources_unchanged_at_end": environment.get("sources_unchanged_at_end"),
        },
    }
    write_json(root / "report-data.json", report)

    return report


def execution_plan(config):
    stages = config["e2e"]["stages"]
    orders = config["e2e"]["orders"]
    devices = config["e2e"]["devices"][: len(stages)]
    assignments = {
        str(stage): devices[index % len(devices)] for index, stage in enumerate(stages)
    }
    affinity = plan_cpu_affinity(
        devices, config["e2e"]["cpu_affinity"], minimum=config["e2e"]["cpu_threads"]
    )

    return {
        "protocol": "Stages run on parallel device lanes; both backends and selected orders of a stage stay on one physical device.",
        "orders": orders,
        "requested_devices": config["e2e"]["devices"],
        "devices": devices,
        "stage_devices": assignments,
        "max_concurrent_workers": len(devices),
        "workers_per_device": 1,
        "expected_workers": len(devices) + len(stages) * 2 * len(orders),
        "cpu_affinity": {str(device): affinity[device] for device in devices},
        "reference_by_device": {
            str(device): reference_path(config, device)
            .relative_to(config["e2e_output_dir"])
            .as_posix()
            for device in devices
        },
    }


def run_lanes(config, environment):
    root = Path(config["e2e_output_dir"])
    environment_path = root / "environment.json"
    identity = environment["run_identity"]
    execution = environment["execution"]
    states = []
    state_lock = threading.Lock()
    active = 0

    def persist():
        write_json(environment_path, environment)
        write_json(
            root / "matrix-status.json", sorted(states, key=lambda item: item["name"])
        )
        build_report(config, environment)

    def lane(device):
        nonlocal active
        stages = [
            stage
            for stage in config["e2e"]["stages"]
            if execution["stage_devices"][str(stage)] == device
        ]
        affinity = execution["cpu_affinity"][str(device)]
        sequence = [("reference", 0, "reference")]

        for selected in config["e2e"]["orders"]:
            order, backends = ORDERS[selected]
            for stage in stages:
                sequence.extend((order, stage, backend) for backend in backends)

        for order, stage, backend in sequence:
            destination = record_path(root, order, stage, backend, device)
            name = (
                f"reference_device-{device}"
                if backend == "reference"
                else f"{order}_stage{stage}_{backend}"
            )
            relative = destination.relative_to(root).as_posix()

            with state_lock:
                if destination.exists():
                    previous = json.loads(destination.read_text(encoding="utf-8"))
                    recorded_hash = environment["result_sha256"].get(relative)

                    if recorded_hash is not None and recorded_hash != sha256_file(
                        destination
                    ):
                        raise ValueError(f"Existing raw result changed: {relative}")

                    if recorded_hash is not None and validate_record(
                        previous, identity, config["e2e"], device, affinity
                    ):
                        states.append(
                            {
                                "name": name,
                                "status": "PASS",
                                "physical_device": device,
                                "reused": True,
                            }
                        )
                        persist()

                        continue

                    history = (
                        root / "attempts" / (name + "-" + str(time.time_ns()) + ".json")
                    )
                    history.parent.mkdir(exist_ok=True)
                    destination.replace(history)
                    previous_log = root / (name + ".log")

                    if previous_log.exists():
                        previous_log.replace(history.with_suffix(".log"))

                    environment["result_sha256"].pop(relative, None)

            command = [
                sys.executable,
                str(Path(__file__).with_name("e2e_runner.py")),
                "--config",
                config["config_file"],
                "--backend",
                backend,
                "--stage",
                str(stage),
                "--order",
                order,
                "--identity",
                identity,
                "--device",
                str(device),
            ]

            if affinity is not None:
                command += ["--cpu-affinity", ",".join(map(str, affinity))]

            print(f"START device={device} {name}", flush=True)

            with state_lock:
                active += 1
                environment["observed_max_concurrent_workers"] = max(
                    environment.get("observed_max_concurrent_workers", 0), active
                )

            try:
                with (root / (name + ".log")).open("w", encoding="utf-8") as log:
                    returncode = run_process(
                        command, log, config["e2e"]["worker_timeout_seconds"]
                    )
            finally:
                with state_lock:
                    active -= 1

            with state_lock:
                if destination.exists():
                    record = json.loads(destination.read_text(encoding="utf-8"))
                else:
                    record = {
                        "status": "ERROR",
                        "run_identity": identity,
                        "error": f"Worker exited {returncode} before saving a result.",
                        "physical_device": device,
                    }

                record["worker_returncode"] = returncode

                if returncode and record["status"] == "PASS":
                    record["status"] = "PROCESS_ERROR_AFTER_RESULT"

                write_json(destination, record)
                environment["result_sha256"][relative] = sha256_file(destination)
                states.append(
                    {
                        "name": name,
                        "status": record["status"],
                        "physical_device": device,
                        "returncode": returncode,
                        "reused": False,
                    }
                )
                persist()

            print(f"{record['status']} device={device} {name}", flush=True)

            if backend == "reference" and record["status"] != "PASS":
                break

    with ThreadPoolExecutor(max_workers=len(execution["devices"])) as executor:
        futures = [executor.submit(lane, device) for device in execution["devices"]]

        for future in as_completed(futures):
            future.result()

    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    root = Path(config["e2e_output_dir"])
    environment_path = root / "environment.json"

    if args.summarize_only:
        environment = json.loads(environment_path.read_text(encoding="utf-8"))

        if config != environment["config"]:
            raise ValueError("Configuration differs from the recorded run.")

        build_report(config, environment)

        return 0

    if environment_path.exists() and not args.resume:
        raise FileExistsError(
            "An E2E run already exists; use --resume or a new output directory."
        )

    root.mkdir(parents=True, exist_ok=True)
    lock_root = Path(config["tmp_dir"]) / "locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    import fcntl

    execution = execution_plan(config)

    with contextlib.ExitStack() as locks:
        output_key = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
        lock_names = [f"ninetoothed-e2e-output-{output_key}.lock"] + [
            f"ninetoothed-e2e-device-{device}.lock"
            for device in sorted(execution["devices"])
        ]

        for name in lock_names:
            lock = locks.enter_context((lock_root / name).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        sources = source_identity(config)
        model = model_identity(config)
        identity = identity_digest(config, sources, model, execution)

        if args.resume and environment_path.exists():
            environment = json.loads(environment_path.read_text(encoding="utf-8"))

            if environment["run_identity"] != identity:
                raise ValueError(
                    "Cannot resume with changed configuration, model or source hashes."
                )

            environment.setdefault("resume_history", []).append(now())
        else:
            environment = {
                "started": now(),
                "run_identity": identity,
                "config": config,
                "sources": sources,
                "model": model,
                "execution": execution,
            }

        environment.setdefault("result_sha256", {})

        for relative, expected_hash in environment["result_sha256"].items():
            path = root / relative

            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"Existing raw result changed: {relative}")

        write_json(environment_path, environment)
        states = run_lanes(config, environment)

        environment.update(
            finished=now(),
            sources_unchanged_at_end=source_identity(config) == sources,
            worker_status_counts=dict(
                collections.Counter(item["status"] for item in states)
            ),
        )
        write_json(environment_path, environment)
        write_json(root / "matrix-status.json", states)
        build_report(config, environment)

        return (
            0
            if environment["sources_unchanged_at_end"]
            and len(states) == execution["expected_workers"]
            and all(item["status"] == "PASS" for item in states)
            else 1
        )


if __name__ == "__main__":
    raise SystemExit(main())
