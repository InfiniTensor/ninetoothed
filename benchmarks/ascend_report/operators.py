"""Measure paired Ascend operator cases from configured, fingerprinted source trees."""

import argparse
import ast
import concurrent.futures
import csv
import datetime
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

if __package__:
    from . import case_profiles
    from .affinity import apply_cpu_affinity, plan_cpu_affinity
    from .operator_adapters import UnsupportedCase, build_call
    from .summarize import _input_contract, _parameters
else:
    import case_profiles
    from affinity import apply_cpu_affinity, plan_cpu_affinity
    from operator_adapters import UnsupportedCase, build_call
    from summarize import _input_contract, _parameters

CONFIG = {}
PATHS = {}
TASK = None
CASES_CSV = {}


def configure(config_file, device=None, devices=None):
    global CONFIG, PATHS, TASK, cases, baseline, CASES_CSV
    config_file = Path(config_file).resolve()
    CONFIG = json.loads(config_file.read_text(encoding="utf-8"))

    def resolve(value):
        path = Path(value).expanduser()
        return (
            (config_file.parent / path).resolve()
            if not path.is_absolute()
            else path.resolve()
        )

    PATHS = {
        name: resolve(CONFIG["paths"][name])
        for name in ("ninetoothed", "ntops", "baseline")
    }
    TASK = resolve(CONFIG["output_dir"]) / "operators"
    CONFIG["cache_dir"] = str(resolve(CONFIG["cache_dir"]))
    CONFIG["tmp_dir"] = str(resolve(CONFIG["tmp_dir"]))
    CONFIG["config_file"] = str(config_file)
    CONFIG.setdefault("operators", {})
    settings = dict(
        case_profile="csv",
        warmup=50,
        device_samples=15,
        api_samples=60,
        window_ms=10,
        timeout_seconds=600,
        devices=CONFIG.get("devices", [0]),
    )
    settings.update(CONFIG["operators"])
    case_profiles.apply_profile([], settings["case_profile"])
    if devices is not None:
        settings["devices"] = devices
    CONFIG["operators"] = settings
    selected_devices = settings["devices"]
    if (
        not selected_devices
        or len(set(selected_devices)) != len(selected_devices)
        or any(type(item) is not int or item < 0 for item in selected_devices)
    ):
        raise ValueError("Specify distinct nonnegative operator device indices.")

    if str(CONFIG.get("scheduling", {}).get("TRITON_ALL_BLOCKS_PARALLEL", "0")) != "0":
        raise ValueError("This report protocol requires TRITON_ALL_BLOCKS_PARALLEL=0.")

    for name in ("warmup", "device_samples", "api_samples", "timeout_seconds"):
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f"Positive measurement setting required: {name}")
    if (
        not isinstance(settings["window_ms"], (int, float))
        or settings["window_ms"] <= 0
    ):
        raise ValueError("Positive window_ms is required.")

    for path in PATHS.values():
        if not path.is_dir():
            raise ValueError(f"Source directory is missing: {path}")
        if TASK.is_relative_to(path):
            raise ValueError("Keep output_dir outside the source checkouts.")

    temporary = Path(CONFIG["tmp_dir"])
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        TMPDIR=str(temporary),
        TMP=str(temporary),
        TEMP=str(temporary),
        TRITON_ALL_BLOCKS_PARALLEL="0",
        PYTHONDONTWRITEBYTECODE="1",
    )
    if device is not None:
        if device not in selected_devices:
            raise ValueError("Worker device must belong to the configured device set.")
        affinity = plan_cpu_affinity(
            selected_devices,
            settings.get("cpu_affinity", CONFIG.get("cpu_affinity", "auto")),
            minimum=1,
        )
        apply_cpu_affinity(affinity[device])
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
        cache = Path(CONFIG["cache_dir"]) / f"device-{device}" / "parallel0" / "triton"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(cache)

    CASES_CSV = {}
    if CONFIG.get("case_csv"):
        path = resolve(CONFIG["case_csv"])
        CONFIG["case_csv"] = str(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = {}
            for column in reader.fieldnames or []:
                match = re.search(r"[（(]([A-Za-z_]+)[）)]$", column)
                columns[column] = match.group(1) if match else column
            for row in reader:
                normalized = {columns[key]: value for key, value in row.items()}
                case_id = normalized["id"]
                if case_id in CASES_CSV:
                    raise ValueError(f"Duplicate CSV case: {case_id}")
                CASES_CSV[case_id] = normalized

    sys.path[:0] = [
        str(PATHS["baseline"] / "benchmarks"),
        str(PATHS["baseline"]),
        str(PATHS["ninetoothed"] / "src"),
        str(PATHS["ntops"] / "src"),
    ]
    cases = importlib.import_module("cases")
    baseline = importlib.import_module("run")
    if (
        Path(cases.__file__).resolve()
        != (PATHS["baseline"] / "benchmarks/cases.py").resolve()
    ):
        raise RuntimeError(
            "An unrelated cases module was already imported; use an isolated process."
        )
    if (
        Path(baseline.__file__).resolve()
        != (PATHS["baseline"] / "benchmarks/run.py").resolve()
    ):
        raise RuntimeError(
            "An unrelated run module was already imported; use an isolated process."
        )


def source_hashes():
    hashes = {}
    for name, part in [("ninetoothed", "src"), ("ntops", "src"), ("baseline", "")]:
        canonical = {"ntops": "ntops.lab", "baseline": "triton-ascend-ops"}.get(
            name, name
        )
        for path in sorted((PATHS[name] / part).rglob("*.py")):
            if "artifacts" not in path.parts and "__pycache__" not in path.parts:
                hashes[canonical + "/" + path.relative_to(PATHS[name]).as_posix()] = (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                )
    return hashes


def source_revisions():
    revisions = {}
    for name, path in PATHS.items():
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=15,
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(path),
                        "status",
                        "--porcelain",
                        "--untracked-files=no",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                ).strip()
            )
            revisions[name] = {"commit": commit, "tracked_changes": dirty}
        except (OSError, subprocess.SubprocessError):
            revisions[name] = {"commit": None, "tracked_changes": None}
    return revisions


def verify_csv_record(record):
    if not CASES_CSV or record["status"] != "PASS":
        return
    original_case = record.get("original_case")
    if record.get("case_profile") == case_profiles.COMMON_CONTRACTS_PROFILE:
        if not isinstance(original_case, dict):
            raise ValueError("Adjusted case has no original-case provenance")
        transformed = case_profiles.apply_profile(
            [original_case], case_profiles.COMMON_CONTRACTS_PROFILE
        )
        if len(transformed) != 1:
            raise ValueError("An excluded case was measured")
        for key, value in transformed[0].items():
            if key != "parameters" and record.get(key) != value:
                raise ValueError(f"Adjusted case differs from profile: {key}")
        shapes, strides = _input_contract(transformed[0])
        if (
            [item["shape"] for item in record["inputs"]] != shapes
            or [item["stride"] for item in record["inputs"]] != strides
            or record["parameters"] != _parameters(transformed[0])
        ):
            raise ValueError("Measured inputs/parameters differ from adjusted plan")
        original_shapes, original_strides = _input_contract(original_case)
        csv_record = dict(
            original_case,
            inputs=[
                {"shape": shape, "stride": stride}
                for shape, stride in zip(original_shapes, original_strides)
            ],
            parameters=_parameters(original_case),
            status="PASS",
        )
        verify_csv_record(csv_record)
        return
    expected = CASES_CSV[record["id"]]
    for key in ("operator", "dtype", "label", "layout"):
        if record[key] != expected[key]:
            raise ValueError(f"CSV mismatch for {record['id']}: {key}")
    measured = {
        "input_shapes": [item["shape"] for item in record["inputs"]],
        "input_strides": [item["stride"] for item in record["inputs"]],
        "parameters": record["parameters"],
    }
    for key in ("input_shapes", "input_strides", "parameters"):
        if measured[key] != json.loads(expected[key]):
            raise ValueError(f"CSV mismatch for {record['id']}: {key}")


def install_measurement_adapter(out):
    source = inspect.getsource(baseline.measure)
    source = source.replace(
        "grid=list(kwargs['grid'])",
        "grid=list(kwargs['grid'](kwargs) if callable(kwargs['grid']) else kwargs['grid'])",
    )
    anchor = "    checks = verify(case, prepared, output, 'public_before_timing')"
    assert source.count(anchor) == 1
    source = source.replace(
        anchor,
        anchor
        + "\n    graph_buffers = prepared['_capture_graph_buffers']() if '_capture_graph_buffers' in prepared else (output if isinstance(output,tuple) else (output,))\n    def graph_result():\n        return prepared['_assemble_graph_result'](graph_buffers) if '_assemble_graph_result' in prepared else output",
    )
    source = source.replace(
        "outputs = output if isinstance(output,tuple) else (output,)",
        "outputs = graph_buffers",
    )
    source = source.replace(
        "verify(case, prepared, output, 'graph_replay_before_timing')",
        "verify(case, prepared, graph_result(), 'graph_replay_before_timing')",
    )
    source = source.replace(
        "verify(case, prepared, output, 'graph_after_timing')",
        "verify(case, prepared, graph_result(), 'graph_after_timing')",
    )
    (out / "measurement_function.py").write_text(source)
    exec(
        compile(source, str(out / "measurement_function.py"), "exec"), baseline.__dict__
    )


def worker(args):
    os.environ.update(
        ASCEND_RT_VISIBLE_DEVICES=str(args.device),
        TRITON_ALL_BLOCKS_PARALLEL="0",
        TRITON_CACHE_DIR=str(
            Path(CONFIG["cache_dir"]) / f"device-{args.device}" / "parallel0" / "triton"
        ),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    import torch
    import torch_npu
    import triton

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.npu.set_device(0)
    out = Path(args.output)
    group = out / args.op
    group.mkdir(exist_ok=True)
    install_measurement_adapter(group)
    import ninetoothed.generation as generation
    from ninetoothed.generation import CodeGenerator
    from ninetoothed.tensor import Tensor

    generation.CACHE_DIR = TASK / "generated-cache" / str(args.device) / args.op
    generation.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    original_attr = CodeGenerator.visit_Attribute
    cast_hits = []

    def visit_attribute(self, node):
        if (
            node.attr == "to"
            and self._in_context(node.value)
            and isinstance(self._context[node.value.id], Tensor)
        ):
            cast_hits.append(node.value.id)
            return ast.copy_location(
                ast.Attribute(
                    value=self._generate_load(self._context[node.value.id]),
                    attr="to",
                    ctx=node.ctx,
                ),
                node,
            )
        return original_attr(self, node)

    CodeGenerator.visit_Attribute = visit_attribute
    jit_module = importlib.import_module("ninetoothed.jit")
    original_import = jit_module.import_from_path
    aliases, modules = [], {}

    def import_alias(name, path):
        mod = original_import(name, path)
        if getattr(mod, "_IS_NPU", False):
            for key, value in list(vars(mod).items()):
                if (
                    key.startswith("launch_")
                    and not key.endswith("_npu")
                    and hasattr(mod, key[7:] + "_npu")
                    and not hasattr(mod, key + "_npu")
                ):
                    setattr(mod, key + "_npu", value)
                    aliases.append(key + "_npu -> " + key)
            modules[str(path)] = mod
        return mod

    jit_module.import_from_path = import_alias
    versions = {}
    for name in ["torch", "torch_npu", "triton-ascend", "numpy", "sympy"]:
        versions[name] = importlib.metadata.version(name)
    runtime = dict(
        python=sys.version,
        platform=platform.platform(),
        versions=versions,
        cann=torch_npu.utils.get_cann_version(),
        triton_import=triton.__version__,
        physical_device=args.device,
        device_name=torch.npu.get_device_name(0),
        device_properties=str(torch.npu.get_device_properties(0)),
    )
    runtime["cpu_affinity"] = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    baseline.write_json(group / "runtime.json", runtime)
    plan = [
        c
        for c in json.loads((out / "plan.json").read_text())
        if c["operator"] == args.op
    ]
    options = SimpleNamespace(
        **{
            key: CONFIG["operators"][key]
            for key in ["warmup", "device_samples", "api_samples", "window_ms"]
        }
    )
    original_prepare = cases.prepare
    original_verify = cases.verify
    for case in plan:
        for impl in ["baseline", "ninetoothed_adapted"]:
            destination = group / (case["id"] + "__" + impl + ".json")
            if destination.exists():
                continue
            baseline.write_json(
                group / "active.json",
                dict(case=case, implementation=impl, started=time.time()),
            )
            print(f"START {case['id']} {impl}", flush=True)
            prep_info, adapter_info = {}, {}

            def prepare(current_case, device):
                p = case_profiles.prepare(current_case, device, original_prepare)
                if "case_profile_metadata" in p:
                    prep_info["case_profile_metadata"] = p["case_profile_metadata"]
                if impl != "baseline":
                    call, info = build_call(current_case, p)
                    for key in ["_capture_graph_buffers", "_assemble_graph_result"]:
                        if key in info:
                            p[key] = info.pop(key)
                    adapter_info.update(info)
                    p["call"] = call
                started = time.perf_counter()
                value = p["call"]()
                torch.npu.synchronize()
                prep_info["first_call_seconds"] = time.perf_counter() - started
                prep_info["checks"] = cases.verify(
                    current_case, p, value, "precompile_validation"
                )
                configs = []
                for mod in modules.values():
                    for name, obj in vars(mod).items():
                        if name.endswith("_with_auto_tuning") and hasattr(
                            obj, "best_config"
                        ):
                            cfg = obj.best_config
                            configs.append(
                                dict(
                                    kernel=name,
                                    kwargs=cfg.kwargs,
                                    num_warps=cfg.num_warps,
                                    num_stages=cfg.num_stages,
                                )
                            )
                prep_info["configs"] = configs
                del value
                return p

            cases.prepare = prepare
            cases.verify = lambda c, p, value, phase: case_profiles.verify(
                c, p, value, phase, original_verify
            )
            fatal = False
            try:
                result = baseline.measure(case, options)
                if impl != "baseline":
                    assert (
                        result["launches_per_call"]
                        == adapter_info["expected_triton_launches"]
                    ), "Unexpected launch count after tuning"
                    assert all(
                        k["name"] != "_matmul_kernel" for k in result["kernels"]
                    ), "Handwritten fallback is not a NineToothed result"
            except UnsupportedCase as error:
                adapter_info.update(error.metadata)
                result = dict(
                    case,
                    status="UNSUPPORTED",
                    error=str(error),
                    error_type=type(error).__name__,
                )
            except Exception as error:
                result = dict(
                    case,
                    status="FAIL",
                    error=str(error),
                    error_type=type(error).__name__,
                    traceback=traceback.format_exc(),
                )
                traceback.print_exc()
                fatal = isinstance(error, RuntimeError)
                try:
                    torch.npu.synchronize()
                except Exception as sync_error:
                    result["context_error"] = str(sync_error)
                    fatal = True
            finally:
                cases.prepare = original_prepare
                cases.verify = original_verify
            result.update(
                implementation=impl,
                runtime=runtime,
                preparation=prep_info,
                adapter=adapter_info,
                compatibility_shim=dict(
                    launcher_aliases=sorted(set(aliases)),
                    parameter_to_lowering=True,
                    parameter_to_hits=len(cast_hits),
                ),
            )
            if impl == "baseline":
                result["compatibility_shim"] = {}
            try:
                verify_csv_record(result)
            except ValueError as error:
                result.update(
                    status="FAIL", error=str(error), error_type="CaseContractMismatch"
                )
            baseline.write_json(destination, result)
            detail = (
                f"device={result['prepared_device']['median_us']:.4f}us api={result['public_api']['median_us']:.2f}us"
                if result["status"] == "PASS"
                else result.get("error_type", "")
            )
            print(f"{result['status']} {case['id']} {impl} {detail}", flush=True)
            if fatal:
                return 75
    baseline.write_json(
        group / "finished.json",
        dict(
            finished=datetime.datetime.now().astimezone().isoformat(),
            generated_sources=list(modules),
        ),
    )
    return 0


def run_card(device, ops, args, out):
    import fcntl

    with open(
        Path(tempfile.gettempdir()) / f"triton_ascend_baseline_npu_{device}.lock", "a"
    ) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        baseline.idle(device)
        for op in ops:
            group = out / op
            group.mkdir(exist_ok=True)
            expected = (
                sum(
                    c["operator"] == op
                    for c in json.loads((out / "plan.json").read_text())
                )
                * 2
            )
            while (
                len(list(group.glob("*__baseline.json")))
                + len(list(group.glob("*__ninetoothed_adapted.json")))
                < expected
            ):
                baseline.idle(device)
                before = len(list(group.glob("*__baseline.json"))) + len(
                    list(group.glob("*__ninetoothed_adapted.json"))
                )
                print(f"CARD {device} START {op}", flush=True)
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    CONFIG["config_file"],
                    "--worker",
                    "--op",
                    op,
                    "--device",
                    str(device),
                    "--output",
                    str(out),
                    "--devices",
                    *[str(item) for item in args.devices],
                ]
                env = os.environ.copy()
                env.update(
                    ASCEND_RT_VISIBLE_DEVICES=str(device),
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    PYTHONDONTWRITEBYTECODE="1",
                )
                log_path = group / f"worker-{int(time.time())}.log"
                (group / "active.json").unlink(missing_ok=True)
                with log_path.open("w") as log:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                        start_new_session=True,
                    )
                    timeout = False
                    while proc.poll() is None:
                        active = group / "active.json"
                        last_started = (
                            json.loads(active.read_text()).get("started", time.time())
                            if active.exists()
                            else log_path.stat().st_mtime
                        )
                        if (
                            time.time() - last_started
                            > CONFIG["operators"]["timeout_seconds"]
                        ):
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                            timeout = True
                            break
                        time.sleep(2)
                rc = proc.returncode
                active = group / "active.json"
                if rc and active.exists():
                    state = json.loads(active.read_text())
                    dest = group / (
                        state["case"]["id"] + "__" + state["implementation"] + ".json"
                    )
                    if not dest.exists():
                        baseline.write_json(
                            dest,
                            dict(
                                state["case"],
                                implementation=state["implementation"],
                                status="FAIL",
                                error_type="WorkerTimeout" if timeout else "WorkerExit",
                                error=f"exit={rc}; see {log_path.name}",
                                physical_device=device,
                            ),
                        )
                after = len(list(group.glob("*__baseline.json"))) + len(
                    list(group.glob("*__ninetoothed_adapted.json"))
                )
                if after <= before:
                    raise RuntimeError(
                        f"{op} made no progress, rc={rc}; see {log_path}"
                    )
            counts = {}
            for p in list(group.glob("*__baseline.json")) + list(
                group.glob("*__ninetoothed_adapted.json")
            ):
                r = json.loads(p.read_text())
                key = r["implementation"] + ":" + r["status"]
                counts[key] = counts.get(key, 0) + 1
            print(f"CARD {device} DONE {op} {counts}", flush=True)
        return baseline.idle(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output")
    p.add_argument("--devices", nargs="+", type=int)
    p.add_argument("--ops", nargs="+")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--op")
    p.add_argument("--device", type=int)
    args = p.parse_args()
    configure(args.config, args.device if args.worker else None, args.devices)
    args.output = args.output or str(TASK)
    args.devices = args.devices or CONFIG["operators"]["devices"]
    if (
        not args.devices
        or len(set(args.devices)) != len(args.devices)
        or any(type(device) is not int or device < 0 for device in args.devices)
    ):
        raise ValueError("Specify distinct nonnegative operator device indices.")
    if args.worker:
        return worker(args)
    import triton_ascend_ops as ta

    ops = args.ops or CONFIG["operators"].get("ops") or sorted(ta.list_operators())
    plan = cases.make_plan(ops)
    if CASES_CSV:
        planned = {case["id"] for case in plan}
        expected = {key for key, row in CASES_CSV.items() if row["operator"] in ops}
        if planned != expected:
            raise ValueError("Selected case plan differs from the supplied CSV")
    original_plan = plan
    profile = CONFIG["operators"]["case_profile"]
    plan = case_profiles.apply_profile(original_plan, profile)
    if not plan:
        raise ValueError("Selected profile has no cases")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=args.resume)
    hashes = source_hashes()
    script_hashes = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in [
            "operators.py",
            "operator_adapters.py",
            "case_profiles.py",
            "fixed_pool.py",
            "affinity.py",
            "summarize.py",
        ]
    }
    meta_path = out / "environment.json"
    if args.resume:
        meta = json.loads(meta_path.read_text())
        assert (
            meta["source_sha256"] == hashes
            and meta["script_sha256"] == script_hashes
            and meta["config"] == CONFIG
        )
        if json.loads((out / "plan.json").read_text()) != plan:
            raise ValueError("Cannot resume with a different case selection.")
        assignment = {int(k): v for k, v in meta["assignment"].items()}
        if set(assignment) != set(args.devices):
            raise ValueError(
                "Cannot resume with a different physical-device assignment."
            )
    else:
        # Each case's two implementations always use the same physical card.
        assignment = {d: [] for d in args.devices}
        loads = {d: 0 for d in args.devices}
        weights = {
            "conv2d": 10,
            "bmm": 12,
            "mm": 8,
            "addmm": 8,
            "scaled_dot_product_attention": 8,
            "reduction": 2,
        }
        cost = {
            op: sum(c["operator"] == op for c in plan) * weights.get(op, 1)
            for op in ops
        }
        for op in sorted(ops, key=lambda x: cost[x], reverse=True):
            d = min(loads, key=loads.get)
            assignment[d].append(op)
            loads[d] += cost[op]
        meta = dict(
            started=datetime.datetime.now().astimezone().isoformat(),
            source_revisions=source_revisions(),
            source_sha256=hashes,
            script_sha256=script_hashes,
            assignment=assignment,
            case_count=len(plan),
            case_profile=profile,
            original_case_count=len(original_plan),
            original_plan_sha256=hashlib.sha256(
                json.dumps(original_plan, sort_keys=True).encode()
            ).hexdigest(),
            operator_count=len(ops),
            measurement={
                key: CONFIG["operators"][key]
                for key in ["warmup", "device_samples", "api_samples", "window_ms"]
            },
            config=CONFIG,
            case_csv_sha256=hashlib.sha256(
                Path(CONFIG["case_csv"]).read_bytes()
            ).hexdigest()
            if CASES_CSV
            else None,
            policy="Existing generated kernels, explicit host adapters and two disclosed code-generation compatibility shims; no new numerical implementations",
            measurement_adapter="Callable grid metadata; RoPE freezes actual kernel output pair and verifies packed result outside device timing; public timing includes packing",
        )
        baseline.write_json(meta_path, meta)
        baseline.write_json(out / "plan.json", plan)
        baseline.write_json(out / "original-plan.json", original_plan)
        if profile != "csv":
            retained = {c.get("original_id", c["id"]) for c in plan}
            baseline.write_json(
                out / "case-adjustments.json",
                dict(
                    case_profile=profile,
                    adjusted_cases=[c for c in plan if c.get("case_profile")],
                    excluded_cases=[
                        c for c in original_plan if c["id"] not in retained
                    ],
                ),
            )
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(assignment)) as pool:
        futures = {
            pool.submit(run_card, d, group, args, out): d
            for d, group in assignment.items()
        }
        for f in concurrent.futures.as_completed(futures):
            d = futures[f]
            meta.setdefault("npu_after", {})[str(d)] = f.result()
    records = []
    for case in plan:
        for impl in ["baseline", "ninetoothed_adapted"]:
            records.append(
                json.loads(
                    (
                        out / case["operator"] / (case["id"] + "__" + impl + ".json")
                    ).read_text()
                )
            )
    baseline.write_json(out / "results.json", records)
    meta.update(
        finished=datetime.datetime.now().astimezone().isoformat(),
        source_unchanged_at_end=source_hashes() == hashes,
    )
    baseline.write_json(meta_path, meta)
    print(f"COMPLETE {len(records)} records", flush=True)


if __name__ == "__main__":
    sys.exit(main())
