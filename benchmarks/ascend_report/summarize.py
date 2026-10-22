"""Audit paired Ascend measurements and write portable operator summaries.

Usage: python summarize.py --input RUN/operators --output RUN

The input directory must contain plan.json, environment.json, and results.json.
Only Python's standard library is required. Raw files are never modified.
Performance percent is baseline median / NineToothed median * 100. Failed,
unsupported, missing, duplicated, or unaudited pairs never enter that ratio.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

if __package__:
    from .case_profiles import apply_profile
else:
    from case_profiles import apply_profile

IMPLEMENTATIONS = ("baseline", "ninetoothed_adapted")
TARGET_CASES = 465
TARGET_OPERATORS = 41
MEASUREMENT_PHASES = frozenset(
    {
        "public_before_timing",
        "graph_replay_before_timing",
        "graph_after_timing",
        "public_after_timing",
    }
)
BOOLEAN_OPS = frozenset("eq ne ge gt le lt isinf isnan".split())
BINARY_OPS = frozenset(
    "add sub mul div pow eq ne ge gt le lt bitwise_and bitwise_or".split()
)
POINTWISE_OPS = frozenset(
    "abs add bitwise_and bitwise_not bitwise_or clamp cos div eq exp ge gelu gt "
    "isinf isnan le lt mul ne neg pow relu rsqrt sigmoid silu sin sub tanh".split()
)
DEFAULT_MEASUREMENT = {"warmup": 50, "device_samples": 15, "api_samples": 60}


def _read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _number(value, *, positive=False):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (not positive or value > 0)
    )


def _integer(value, *, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _stats(samples):
    ordered = sorted(samples)
    position = (len(ordered) - 1) * 0.95
    low, high = math.floor(position), math.ceil(position)
    mean = statistics.mean(samples)

    return {
        "median_us": statistics.median(samples),
        "p95_us": ordered[low] + (ordered[high] - ordered[low]) * (position - low),
        "min_us": min(samples),
        "max_us": max(samples),
        "mean_us": mean,
        "cv_pct": 100 * statistics.pstdev(samples) / mean,
    }


def _audit_stats(record, key, count):
    saved = record.get(key)
    if not isinstance(saved, dict):
        return [f"{key}: statistics missing"], None
    samples = saved.get("samples_us")
    if not isinstance(samples, list) or len(samples) != count:
        return [f"{key}: expected {count} raw samples"], None
    if not all(_number(value, positive=True) for value in samples):
        return [f"{key}: samples must be finite and strictly positive"], None
    computed = _stats(samples)
    errors = []
    for field, expected in computed.items():
        value = saved.get(field)
        if not _number(value) or not math.isclose(
            value, expected, rel_tol=1e-9, abs_tol=1e-8
        ):
            errors.append(f"{key}.{field}: saved={value!r}, recomputed={expected!r}")

    return errors, computed


def _strides(shape, transpose=False):
    sizes = list(shape)
    if transpose and len(sizes) >= 2:
        sizes[-2], sizes[-1] = sizes[-1], sizes[-2]
    result, stride = [], 1
    for size in reversed(sizes):
        result.append(stride)
        stride *= size
    result.reverse()
    if transpose and len(sizes) >= 2:
        result[-2], result[-1] = result[-1], result[-2]

    return result


def _input_contract(case):
    """Derive public input metadata from the immutable benchmark plan."""
    op, layout = case["operator"], case["layout"]
    transpose = layout == "transpose"
    shapes, transposed = [], []

    def tensor(shape, apply_layout=False):
        shapes.append(list(shape))
        transposed.append(transpose and apply_layout)

    if op in POINTWISE_OPS:
        tensor(case["shape"], True)
        if op in BINARY_OPS:
            tensor(
                [1, case["shape"][-1]] if layout == "broadcast" else case["shape"],
                layout != "broadcast",
            )
    elif op in {"mm", "addmm", "bmm"}:
        m, k, n = case["dims"]
        if op == "addmm":
            tensor([n])
        prefix = [case["batch"]] if op == "bmm" else []
        tensor(prefix + [m, k], True)
        tensor(prefix + [k, n], True)
    elif op in {"rms_norm", "layer_norm"}:
        tensor(case["shape"], True)
        tensor([case["shape"][-1]])
        if op == "layer_norm":
            tensor([case["shape"][-1]])
    elif op == "rotary_position_embedding":
        tensor(case["shape"], True)
        tensor([case["shape"][0], case["shape"][-1] // 2])
        tensor([case["shape"][0], case["shape"][-1] // 2])
    elif op == "scaled_dot_product_attention":
        batch, qh, kh, sq, sk, width = case["dims"]
        tensor([batch, qh, sq, width], True)
        tensor([batch, kh, sk, width], True)
        tensor([batch, kh, sk, width], True)
    elif op == "conv2d":
        tensor(case["shape"], True)
        tensor(case["weight_shape"])
        if not case.get("no_bias"):
            tensor([case["weight_shape"][0]])
    elif op in {"softmax", "swiglu", "reduction", "max_pool2d", "dropout"}:
        tensor(case["shape"], True)
    else:
        raise ValueError(f"No input contract for operator {op!r}")

    return shapes, [_strides(s, t) for s, t in zip(shapes, transposed)]


def _parameters(case):
    op, parameters = case["operator"], dict(case.get("parameters", {}))
    if op == "clamp":
        if case.get("case_profile") == "common_contracts_v1":
            return {"min": parameters["min"], "max": parameters["max"]}
        return {"min": -0.25, "max": 0.75}
    if op == "addmm":
        return {
            "alpha": parameters.get("alpha", 1.0),
            "beta": parameters.get("beta", 1.0),
        }
    if op in {"layer_norm", "rms_norm"}:
        return {"eps": 1e-5}
    if op == "softmax":
        parameters.setdefault("dim", -1)
    if op == "rotary_position_embedding":
        parameters.setdefault("interleaved", False)
    if op == "dropout":
        if case.get("case_profile") == "common_contracts_v1":
            return {
                "p": parameters["p"],
                "training": parameters["training"],
                "return_mask": False,
            }
        return {
            "p": parameters.get("p", 0.25),
            "training": parameters.get("training", True),
            "seed": 123,
            "offset": 17,
            "return_mask": True,
        }

    return parameters


def _uuid(runtime):
    value = runtime.get("physical_device_uuid") or runtime.get("device_uuid")
    if value:
        return str(value).lower()
    properties = runtime.get("device_properties", runtime.get("properties", ""))
    if isinstance(properties, dict):
        value = properties.get("uuid")
        return str(value).lower() if value else None
    match = re.search(r"\buuid\s*=\s*['\"]?([^,'\")\s]+)", str(properties))

    return match.group(1).lower() if match else None


def _physical(record):
    return (record.get("runtime") or {}).get(
        "physical_device", record.get("physical_device")
    )


def _check_correctness(checks, phases, label):
    if not isinstance(checks, list) or not checks:
        return [f"{label}: correctness records missing"]
    if not all(isinstance(check, dict) for check in checks):
        return [f"{label}: correctness records must be objects"]
    errors = []
    found = {check.get("phase") for check in checks}
    if not phases <= found:
        errors.append(f"{label}: missing phases {sorted(phases - found)}")
    for check in checks:
        if check.get("passed") is not True:
            errors.append(f"{label}: check did not pass")
        if "mismatch_count" in check and check["mismatch_count"] != 0:
            errors.append(f"{label}: nonzero mismatch_count in PASS record")
        if "max_abs_error" in check and not _number(check["max_abs_error"]):
            errors.append(f"{label}: nonfinite max_abs_error")

    return errors


def _check_tolerances(checks, case):
    dtype, op = case["dtype"], case["operator"]
    rtol = {"bfloat16": 0.025, "float16": 0.004}.get(dtype, 0.0005)
    atol = rtol
    if op in {"mm", "addmm", "bmm", "conv2d"}:
        atol = {"bfloat16": 0.08, "float16": 0.01}.get(dtype, 0.003)
    if op == "scaled_dot_product_attention":
        atol = 0.015 if dtype == "bfloat16" else 0.001
    errors = []
    for check in checks or []:
        if not isinstance(check, dict):
            continue
        if "rtol" not in check and "atol" not in check:
            continue
        exact = (
            op in BOOLEAN_OPS
            or dtype == "int32"
            or check.get("name")
            in {
                "indices",
                "fixed_seed_mask",
            }
        )
        expected = (0, 0) if exact else (rtol, atol)
        if (check.get("rtol"), check.get("atol")) != expected:
            errors.append("Correctness tolerances differ from original case contract")

    return errors


def _audit_record(record, case, measurement, assignments):
    errors = []
    for field in (
        "id",
        "operator",
        "dtype",
        "label",
        "layout",
        "shape",
        "dims",
        "batch",
        "weight_shape",
        "no_bias",
        "case_profile",
        "original_id",
        "original_case",
        "original_case_sha256",
    ):
        if field in case and record.get(field) != case[field]:
            errors.append(f"{field}: differs from plan")
    status = record.get("status")
    if status not in {"PASS", "FAIL", "UNSUPPORTED", "PROCESS_ERROR_AFTER_CHECK"}:
        errors.append(f"Unknown status {status!r}")
    expected_device = assignments.get(case["operator"])
    if expected_device is not None and _physical(record) != expected_device:
        errors.append("Physical device differs from operator assignment")
    if status != "PASS":
        return {"pass_audited": False, "errors": errors}

    errors += _check_correctness(record.get("checks"), MEASUREMENT_PHASES, "checks")
    preparation = record.get("preparation") or {}
    prechecks = preparation.get("checks")
    if prechecks is None:
        prechecks = preparation.get("precompile_checks")
    prephases = {c.get("phase") for c in prechecks or [] if isinstance(c, dict)}
    expected_phase = (
        "precompile_validation"
        if "precompile_validation" in prephases
        else "precompile"
    )
    errors += _check_correctness(prechecks, {expected_phase}, "precompile")
    errors += _check_tolerances(record.get("checks"), case)
    errors += _check_tolerances(prechecks, case)
    for field in ("device_samples", "api_samples"):
        if record.get(field) != measurement[field]:
            errors.append(f"Declared {field} differs from environment")
    if record.get("warmup_per_path") != measurement["warmup"]:
        errors.append("Declared warmup count differs from environment")
    d_errors, device = _audit_stats(
        record, "prepared_device", measurement["device_samples"]
    )
    a_errors, api = _audit_stats(record, "public_api", measurement["api_samples"])
    errors += d_errors + a_errors
    inputs = record.get("inputs")
    shapes, strides = _input_contract(case)
    if not isinstance(inputs, list) or not all(isinstance(x, dict) for x in inputs):
        errors.append("Input metadata missing or malformed")
    else:
        if [x.get("shape") for x in inputs] != shapes:
            errors.append("Input shapes differ from plan")
        if [x.get("stride") for x in inputs] != strides:
            errors.append("Input strides differ from plan")
        if any(
            str(x.get("dtype", "")).removeprefix("torch.") != case["dtype"]
            for x in inputs
        ):
            errors.append("Input dtypes differ from plan")
    if record.get("parameters") != _parameters(case):
        errors.append("Prepared parameters differ from plan contract")
    runtime = record.get("runtime") or {}
    versions = runtime.get("versions") or {}
    for key in ("torch", "torch_npu", "triton-ascend"):
        if not versions.get(key):
            errors.append(f"Runtime version missing: {key}")
    if not runtime.get("cann") or not runtime.get("triton_import"):
        errors.append("CANN/imported Triton version missing")
    if not _integer(_physical(record)) or not _uuid(runtime):
        errors.append("Physical device number or UUID missing")
    kernels, launches = record.get("kernels"), record.get("launches_per_call")
    if not _integer(launches, minimum=1):
        errors.append("Invalid launches_per_call")
    if not isinstance(kernels, list) or len(kernels) != launches:
        errors.append("Captured kernel count differs from launches_per_call")
    elif any(k.get("compiled_type") != "CompiledKernel" for k in kernels):
        errors.append("Captured object is not a CompiledKernel")
    if record.get("implementation") == "ninetoothed_adapted":
        adapter = record.get("adapter") or {}
        expected = case["batch"] if case["operator"] == "bmm" else 1
        if launches != expected or adapter.get("expected_triton_launches") != expected:
            errors.append(f"Expected {expected} generated launches per operation")
        if adapter.get("implementation_type") != "ninetoothed_generated":
            errors.append("Adapter does not identify existing generated kernels")
        if adapter.get("output_contract") != "tensor":
            errors.append("Public output contract is not a complete tensor")
        if isinstance(kernels, list) and any(
            k.get("name") == "_matmul_kernel" for k in kernels
        ):
            errors.append("Handwritten matrix fallback entered generated series")
        if (
            case["operator"] == "rotary_position_embedding"
            and not _parameters(case)["interleaved"]
        ):
            if (
                adapter.get("graph_validation_protocol")
                != "freeze_kernel_pair_then_assemble_full_tensor_outside_device_timing"
            ):
                errors.append("Split RoPE graph-buffer validation protocol missing")
    nodes = record.get("graph_nodes")
    if not _integer(nodes, minimum=1) or not _integer(
        record.get("graph_repeats_per_sample"), minimum=1
    ):
        errors.append("Invalid graph invocation/replay counts")

    return {
        "pass_audited": not errors,
        "errors": errors,
        "device": device,
        "api": api,
        "physical_uuid": _uuid(runtime),
        "actual_triton_graph_launches": nodes * launches
        if _integer(nodes, minimum=1) and _integer(launches, minimum=1)
        else None,
    }


def _audit_pair(baseline, nine):
    errors = []
    br, nr = baseline.get("runtime") or {}, nine.get("runtime") or {}
    if _physical(baseline) != _physical(nine):
        errors.append("Pair physical device numbers differ")
    if not _uuid(br) or _uuid(br) != _uuid(nr):
        errors.append("Pair physical UUIDs differ or are missing")
    for field in ("versions", "cann", "triton_import", "python", "cpu_affinity"):
        if br.get(field) != nr.get(field):
            errors.append(f"Pair runtime differs: {field}")
    for field in ("inputs", "parameters"):
        if baseline.get(field) != nine.get(field):
            errors.append(f"Pair metadata differs: {field}")
    compiler_environments = [
        {
            key: value
            for key, value in (record.get("compiler_environment") or {}).items()
            if key not in {"TRITON_CACHE_DIR", "TMPDIR"}
        }
        for record in (baseline, nine)
    ]
    if compiler_environments[0] != compiler_environments[1]:
        errors.append("Pair compiler environment flags differ")

    return errors


def _diagnostic(record):
    if record is None:
        return "missing_measurement", "Measurement record missing", ""
    if record.get("status") == "PASS":
        return "", "", ""
    error = str(record.get("error") or "")
    trace = str(record.get("traceback") or "")
    text = error + "\n" + trace
    if record.get("status") == "UNSUPPORTED":
        return "unsupported_contract", error, error
    if "No valid triton configs" in text:
        first = next(
            line.strip()
            for line in text.splitlines()
            if "No valid triton configs" in line
        )
        return "autotune_no_valid_configuration", first, error
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    final = next(
        (
            line
            for line in reversed(lines)
            if re.match(r"[\w.]+(?:Error|Exception|Failure):", line)
        ),
        lines[-1] if lines else str(record.get("error_type", "Unknown failure")),
    )
    if "upper_bound" in text:
        category = "codegen_missing_symbol_bound"
    elif re.search(r"Cannot broadcast|broadcast.*expanded size", text, re.I):
        category = "compiler_tile_broadcast"
    elif re.search(
        r"\bUB\b.*(?:overflow|exceed)|ub overflow|unified[ _-]*buffer|out.?of.?resources",
        text,
        re.I,
    ):
        category = "compiler_local_memory_resource"
    elif "mismatch_count" in text:
        category = "correctness_mismatch"
    elif "Timeout" in str(record.get("error_type")):
        category = "worker_timeout"
    elif "WorkerExit" in str(record.get("error_type")):
        category = "worker_exit"
    elif re.search(r"Compilation|MLIR|compiler", text, re.I):
        category = "compiler_failure"
    else:
        category = "runtime_or_source_failure"

    return category, final, error


def _geo(values):
    return (
        math.exp(statistics.mean(math.log(value) for value in values))
        if values
        else None
    )


def _metrics(pairs, operators):
    ratios = [p["ratio"] for p in pairs]
    grouped = collections.defaultdict(list)
    for pair in pairs:
        grouped[pair["operator"]].append(pair["ratio"])
    bsum = math.fsum(p["baseline_us"] for p in pairs)
    nsum = math.fsum(p["nine_us"] for p in pairs)

    return {
        "valid_cases": len(pairs),
        "target_cases": TARGET_CASES,
        "operators_with_valid_pairs": len(grouped),
        "target_operators": TARGET_OPERATORS,
        "case_geomean_percent": 100 * _geo(ratios) if ratios else None,
        "operator_equal_geomean_percent": 100
        * _geo([_geo(v) for v in grouped.values()])
        if grouped
        else None,
        "case_arithmetic_mean_ratio_percent": 100 * statistics.mean(ratios)
        if ratios
        else None,
        "operator_equal_arithmetic_mean_ratio_percent": 100
        * statistics.mean(statistics.mean(v) for v in grouped.values())
        if grouped
        else None,
        "sum_baseline_us": bsum if pairs else None,
        "sum_ninetoothed_us": nsum if pairs else None,
        "ratio_of_summed_times_percent": 100 * bsum / nsum if pairs else None,
        "fully_covered_operators": sum(op["full_coverage"] for op in operators),
    }


def _thresholds(operators):
    complete = [op for op in operators if op["full_coverage"]]
    subset = [op for op in operators if op["valid_cases"]]
    output = {
        "fully_covered_operator_count": len(complete),
        "target_operator_count": TARGET_OPERATORS,
    }
    for threshold in (85, 95):
        output[f"ge_{threshold}_percent"] = {
            "complete_operators_meeting_threshold": sum(
                op["formal_operator_geomean_percent"] >= threshold for op in complete
            ),
            "complete_operator_denominator": len(complete),
            "all_operator_denominator": TARGET_OPERATORS,
            "subset_operators_meeting_threshold": sum(
                op["partial_case_geomean_percent"] >= threshold for op in subset
            ),
            "subset_operator_denominator": len(subset),
            "subset_count_is_not_full_operator_certification": True,
        }

    return output


def _make_row(case, records, audits, pair_errors):
    shapes, strides = _input_contract(case)
    row = {
        "id": case["id"],
        "operator": case["operator"],
        "dtype": case["dtype"],
        "label": case["label"],
        "layout": case["layout"],
        "input_shapes": json.dumps(shapes),
        "input_strides": json.dumps(strides),
        "parameters": json.dumps(_parameters(case), sort_keys=True),
        "valid_paired_comparison": not pair_errors
        and all(audits[impl]["pass_audited"] for impl in IMPLEMENTATIONS),
        "performance_percent": None,
        "api_performance_percent": None,
        "pair_audit_errors": json.dumps(pair_errors, ensure_ascii=False),
    }
    for impl, prefix in (
        ("baseline", "baseline"),
        ("ninetoothed_adapted", "ninetoothed"),
    ):
        record, audit = records[impl], audits[impl]
        category, diagnostic, error = _diagnostic(record)
        row.update(
            {
                prefix + "_status": record.get("status") if record else "MISSING",
                prefix + "_audit_pass": audit["pass_audited"],
                prefix + "_audit_errors": json.dumps(
                    audit["errors"], ensure_ascii=False
                ),
                prefix + "_physical_device": _physical(record) if record else None,
                prefix + "_physical_uuid": _uuid(record.get("runtime") or {})
                if record
                else None,
                prefix + "_error_type": record.get("error_type", "") if record else "",
                prefix + "_category": category,
                prefix + "_final_diagnostic": diagnostic,
                prefix + "_raw_error": error,
                prefix + "_launches_per_call": record.get("launches_per_call")
                if record
                else None,
                prefix + "_graph_logical_invocations": record.get("graph_nodes")
                if record
                else None,
                prefix + "_actual_triton_graph_launches": audit.get(
                    "actual_triton_graph_launches"
                ),
            }
        )
        for scope in ("device", "api"):
            stats = audit.get(scope) or {}
            for field in ("median_us", "p95_us", "cv_pct"):
                row[f"{prefix}_{scope}_{field}"] = stats.get(field)
    nine = records["ninetoothed_adapted"] or {}
    adapter = nine.get("adapter") or {}
    row.update(
        {
            "adapter_mode": adapter.get("mode", "adapted"),
            "adapter_native_contract_match": adapter.get("native_contract_match"),
            "adapter_kinds": ";".join(
                sorted({v.get("kind", "") for v in adapter.get("adaptations", [])})
            ),
            "adapter_details": json.dumps(
                adapter.get("adaptations", []), ensure_ascii=False
            ),
            "numeric_notes": json.dumps(
                adapter.get("numeric_notes", []), ensure_ascii=False
            ),
            "graph_validation_protocol": adapter.get(
                "graph_validation_protocol", "direct_output_buffer"
            ),
            "compatibility_shim": json.dumps(
                nine.get("compatibility_shim", {}), ensure_ascii=False
            ),
        }
    )
    if row["valid_paired_comparison"]:
        row["performance_percent"] = (
            100
            * audits["baseline"]["device"]["median_us"]
            / audits["ninetoothed_adapted"]["device"]["median_us"]
        )
        row["api_performance_percent"] = (
            100
            * audits["baseline"]["api"]["median_us"]
            / audits["ninetoothed_adapted"]["api"]["median_us"]
        )

    return row


def _operator_table(rows):
    operators = []
    for name in sorted({row["operator"] for row in rows}):
        group = [row for row in rows if row["operator"] == name]
        valid = [row for row in group if row["valid_paired_comparison"]]
        full = len(valid) == len(group)
        mean = _geo([row["performance_percent"] for row in valid])
        failures = collections.Counter()
        for row in group:
            if row["valid_paired_comparison"]:
                continue
            if row["ninetoothed_status"] != "PASS":
                category = row["ninetoothed_category"]
            elif row["baseline_status"] != "PASS":
                category = "baseline_" + row["baseline_category"]
            else:
                category = "audit_invalid"
            failures[category] += 1
        operators.append(
            {
                "operator": name,
                "valid_cases": len(valid),
                "target_cases": len(group),
                "full_coverage": full,
                "coverage_status": "COMPLETE"
                if full
                else "PARTIAL"
                if valid
                else "NO_VALID_PAIRS",
                "partial_case_geomean_percent": mean,
                "formal_operator_geomean_percent": mean if full else None,
                "baseline_status_counts": dict(
                    collections.Counter(row["baseline_status"] for row in group)
                ),
                "ninetoothed_status_counts": dict(
                    collections.Counter(row["ninetoothed_status"] for row in group)
                ),
                "invalid_pair_categories": dict(failures),
                "adapter_kinds": sorted(
                    {
                        kind
                        for row in group
                        for kind in row["adapter_kinds"].split(";")
                        if kind
                    }
                ),
            }
        )

    return operators


def _percentage(value):
    return "—" if value is None else f"{value:.2f}%"


def _markdown(summary):
    valid = summary["valid_subset"]
    coverage = summary["coverage"]
    lines = [
        "# 昇腾算子配对性能统计",
        "",
        f"目标41类算子、465个case，共930条路径；收到 {coverage['observed_expected_paths']}/930 条唯一目标记录。",
        "",
        f"有效配对 **{valid['valid_cases']}/465**，有有效结果的算子 **{valid['operators_with_valid_pairs']}/41**，完整覆盖算子 **{valid['fully_covered_operators']}/41**。",
        "",
        f"完整41类/465case正式指标：**{summary['formal_full_matrix']['status']}**。FAIL和UNSUPPORTED不进入性能比率，部分算子子集不能替代完整目标。",
        "",
        "相对性能 = 同物理卡和UUID的baseline设备中位耗时 / 九齿中位耗时 ×100%，越大越快。",
        "",
        f"有效子集case几何均值 {_percentage(valid['case_geomean_percent'])}；有结果算子等权几何均值 {_percentage(valid['operator_equal_geomean_percent'])}；case比率算术均值 {_percentage(valid['case_arithmetic_mean_ratio_percent'])}；合计时间比 {_percentage(valid['ratio_of_summed_times_percent'])}。四者权重不同，不可互换。",
        "",
        f"有效子集baseline耗时合计 {valid['sum_baseline_us']} μs；九齿 {valid['sum_ninetoothed_us']} μs。此合计每case各一次，不是实际模型端到端耗时。",
        "",
        "使用显式host布局/参数适配和已披露编译兼容，不称未经适配的原始公开API成绩。",
        "",
    ]
    if summary.get("case_profile") == "common_contracts_v1":
        lines[2] = (
            f"调整测例范围41类算子、453个case，共906条路径；收到 {coverage['observed_expected_paths']} 条唯一目标记录。"
        )
        lines[4] = (
            f"有效配对 **{valid['valid_cases']}/453**，有有效结果的算子 **{valid['operators_with_valid_pairs']}/41**，完整覆盖算子 **{valid['fully_covered_operators']}/41**。"
        )
        lines[6] = (
            f"调整范围指标：**{summary['adjusted_full_matrix']['status']}**；原CSV41类/465case指标仍为**NOT_ESTABLISHED**。clamp[-1,1]；max_pool2d仅值；dropout仅推理/p=0；reduction排除prod；addmm使用单位缩放；conv2d使用普通卷积。"
        )
    for threshold in (85, 95):
        count = summary["operator_threshold_counts"][f"ge_{threshold}_percent"]
        lines.append(
            f"- ≥{threshold}%：完整覆盖算子中 {count['complete_operators_meeting_threshold']}/{count['complete_operator_denominator']}（全部目标分母41）；含部分覆盖的子集 {count['subset_operators_meeting_threshold']}/{count['subset_operator_denominator']}，后者不代表完整算子达标。"
        )
    lines += [
        "",
        "| 算子 | 有效/目标 | 有效配对几何均值 | 覆盖/失败 |",
        "|---|---:|---:|---|",
    ]
    for op in summary["operators"]:
        note = (
            "完整"
            if op["full_coverage"]
            else "部分子集"
            if op["valid_cases"]
            else "无有效配对"
        )
        if op["invalid_pair_categories"]:
            note += "；" + "; ".join(
                f"{k}:{v}" for k, v in op["invalid_pair_categories"].items()
            )
        lines.append(
            f"| {op['operator']} | {op['valid_cases']}/{op['target_cases']} | {_percentage(op['partial_case_geomean_percent'])} | {note} |"
        )
    lines += [
        "",
        f"数据审计：{'通过' if summary['audit_passed'] else '未通过'}，{len(summary['audit_errors'])}项错误。",
        "",
        "PASS逐项检查预编译和四个测量验证阶段、有限正原始样本及统计复算、输入形状/步长/参数、同卡UUID及版本。Graph节点数按逻辑算子调用计；BMM实际Triton launch数为graph_nodes×launches_per_call。",
        "",
        "RoPE公开API包含输出打包；设备计时冻结真实输出pair，回放后在计时外打包并验证完整结果。",
    ]
    if summary["audit_errors"]:
        lines += ["", "审计错误（前20项）："]
        lines += ["- " + error for error in summary["audit_errors"][:20]]

    return "\n".join(lines) + "\n"


def aggregate(input_dir, output_dir=None, *, require_full_plan=True):
    """Audit a result directory, write three artifacts, and return its summary.

    Audit failures are reported in the returned object and exclude affected
    comparisons. Malformed input JSON or an invalid plan raises ValueError.
    Experimental FAIL/UNSUPPORTED outcomes are not themselves audit errors.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir.parent
    plan = _read(input_dir / "plan.json")
    environment = _read(input_dir / "environment.json")
    records = _read(input_dir / "results.json")
    if (
        not isinstance(plan, list)
        or not plan
        or not all(isinstance(c, dict) and isinstance(c.get("id"), str) for c in plan)
    ):
        raise ValueError("plan.json must be a nonempty list of cases with string IDs")
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise ValueError("results.json must be a list of record objects")
    if not isinstance(environment, dict):
        raise ValueError("environment.json must be an object")
    errors = []
    profile = environment.get("case_profile", "csv")
    if profile not in {"csv", "common_contracts_v1"}:
        raise ValueError(f"Unknown case profile: {profile!r}")
    target_cases = 453 if profile == "common_contracts_v1" else TARGET_CASES
    if profile == "csv" and any(case.get("case_profile") for case in plan):
        errors.append("Adjusted cases cannot establish the original CSV scope")
    if profile == "common_contracts_v1":
        original_file = input_dir / "original-plan.json"
        if not original_file.is_file():
            errors.append("Adjusted profile requires the original plan")
        else:
            original_plan = _read(original_file)
            if apply_profile(original_plan, profile) != plan:
                errors.append("Adjusted plan differs from the declared profile")
            digest = hashlib.sha256(
                json.dumps(original_plan, sort_keys=True).encode()
            ).hexdigest()
            if environment.get("original_plan_sha256") != digest:
                errors.append("Original plan provenance digest differs")
    counts = collections.Counter(case["id"] for case in plan)
    if any(count != 1 for count in counts.values()):
        raise ValueError("plan.json contains duplicate case IDs")
    plan_complete = (
        len(plan) == target_cases
        and len({case["operator"] for case in plan}) == TARGET_OPERATORS
    )
    if require_full_plan and not plan_complete:
        errors.append(
            f"Plan does not cover all 41 operators / {target_cases} target cases"
        )
    for field, expected in (
        ("case_count", len(plan)),
        ("operator_count", len({c["operator"] for c in plan})),
    ):
        if field in environment and environment[field] != expected:
            errors.append(f"Environment {field} differs from plan")
    if environment.get("source_unchanged_at_end") is False:
        errors.append("Source files changed during measurement")
    measurement = dict(DEFAULT_MEASUREMENT)
    for field in measurement:
        if field in environment.get("measurement", {}):
            measurement[field] = environment["measurement"][field]
        if not _integer(measurement[field], minimum=0 if field == "warmup" else 1):
            raise ValueError(f"Invalid measurement setting: {field}")
    assignments = {}
    for device, names in environment.get("assignment", {}).items():
        for name in names:
            if name in assignments:
                errors.append(f"Duplicate operator device assignment: {name}")
            assignments[name] = int(device)
    cases = {case["id"]: case for case in plan}
    expected_keys = {(cid, impl) for cid in cases for impl in IMPLEMENTATIONS}
    indexed, duplicate, unknown = {}, [], []
    for record in records:
        key = (record.get("id"), record.get("implementation"))
        if key not in expected_keys:
            unknown.append(list(key))
        elif key in indexed:
            duplicate.append(list(key))
        else:
            indexed[key] = record
    missing = sorted(expected_keys - set(indexed))
    if duplicate:
        errors.append(f"{len(duplicate)} duplicate measurement paths")
    if unknown:
        errors.append(f"{len(unknown)} unknown measurement paths")
    if missing:
        errors.append(f"{len(missing)} missing measurement paths")
    duplicated_keys = {tuple(key) for key in duplicate}
    audits = {}
    for key, record in indexed.items():
        audit = _audit_record(record, cases[key[0]], measurement, assignments)
        if key in duplicated_keys:
            audit["errors"].append("Duplicate measurement path")
            audit["pass_audited"] = False
        audits[key] = audit
        errors += [f"{key[0]}/{key[1]}: {message}" for message in audit["errors"]]
    rows, pairs = [], []
    for case in plan:
        selected = {impl: indexed.get((case["id"], impl)) for impl in IMPLEMENTATIONS}
        checked = {
            impl: audits.get(
                (case["id"], impl),
                {"pass_audited": False, "errors": ["Measurement missing"]},
            )
            for impl in IMPLEMENTATIONS
        }
        pair_errors = []
        if all(
            record and record.get("status") == "PASS" for record in selected.values()
        ):
            pair_errors = _audit_pair(
                selected["baseline"], selected["ninetoothed_adapted"]
            )
            errors += [f"{case['id']} pair: {message}" for message in pair_errors]
        if environment.get("source_unchanged_at_end") is False:
            pair_errors.append("Source files changed during measurement")
        row = _make_row(case, selected, checked, pair_errors)
        rows.append(row)
        if row["valid_paired_comparison"]:
            b, n = (
                checked["baseline"]["device"]["median_us"],
                checked["ninetoothed_adapted"]["device"]["median_us"],
            )
            pairs.append(
                {
                    "id": case["id"],
                    "operator": case["operator"],
                    "ratio": b / n,
                    "baseline_us": b,
                    "nine_us": n,
                }
            )
    operators = _operator_table(rows)
    subset = _metrics(pairs, operators)
    subset["target_cases"] = target_cases
    complete = (
        not errors
        and len(pairs) == target_cases
        and len(operators) == TARGET_OPERATORS
        and all(op["full_coverage"] for op in operators)
    )
    formal = {
        "status": "ESTABLISHED" if complete and profile == "csv" else "NOT_ESTABLISHED",
        "target_cases": TARGET_CASES,
        "target_operators": TARGET_OPERATORS,
    }
    formal.update(
        {
            key: value if complete and profile == "csv" else None
            for key, value in subset.items()
            if key.endswith("percent") or key.startswith("sum_")
        }
    )
    summary = {
        "schema_version": 2,
        "case_profile": profile,
        "measurement_identity": "Existing generated kernels with explicit host adapters and disclosed compiler compatibility; not pristine public APIs",
        "performance_definition": "paired baseline device median_us / ninetoothed device median_us * 100; larger is better",
        "metric_definitions": {
            "case_geomean": "100*exp(mean(log(baseline_us/ninetoothed_us))) over audited valid pairs",
            "operator_equal_geomean": "Geometric mean of each represented operator's valid-case geometric mean; equal operator weights",
            "case_arithmetic_mean": "100*mean(baseline_us/ninetoothed_us) over audited valid pairs",
            "ratio_of_summed_times": "100*sum(baseline_us)/sum(ninetoothed_us) over the same valid pairs; not an end-to-end workload",
        },
        "coverage": {
            "target_operators": TARGET_OPERATORS,
            "target_cases": target_cases,
            "target_paths": target_cases * 2,
            "plan_operators": len(operators),
            "plan_cases": len(plan),
            "plan_covers_full_target": plan_complete,
            "all_selected_paths_present_once": not missing
            and not duplicate
            and not unknown
            and len(records) == len(expected_keys),
            "raw_record_count": len(records),
            "observed_expected_paths": len(indexed),
            "missing_paths": missing,
            "duplicate_paths": duplicate,
            "unknown_paths": unknown,
            "all_expected_paths_present_once": not missing
            and not duplicate
            and not unknown
            and len(records) == target_cases * 2
            and len(plan) == target_cases,
        },
        "valid_subset": subset,
        "formal_full_matrix": formal,
        "operators": operators,
        "operator_threshold_counts": _thresholds(operators),
        "measurement_status_counts": {
            impl: dict(
                collections.Counter(
                    record.get("status", "MISSING")
                    for (cid, name), record in indexed.items()
                    if name == impl
                )
            )
            for impl in IMPLEMENTATIONS
        },
        "audit_passed": not errors,
        "audit_errors": errors,
        "measurement": measurement,
        "provenance": {
            "input_files_sha256": {
                name: _sha(input_dir / name)
                for name in ("plan.json", "environment.json", "results.json")
            },
            "aggregator_sha256": _sha(Path(__file__)),
            "recorded_source_sha256": environment.get("source_sha256", {}),
            "recorded_script_sha256": environment.get("script_sha256", {}),
            "source_unchanged_at_end": environment.get("source_unchanged_at_end"),
            "source_hash_note": "Recorded manifests are preserved; this portable data audit does not require external source checkouts.",
        },
        "graph_notes": "graph_nodes counts logical calls; actual_triton_graph_launches=graph_nodes*launches_per_call, including all BMM batch kernels",
    }
    if profile == "common_contracts_v1":
        summary["adjusted_full_matrix"] = {
            "case_profile": profile,
            "status": "ESTABLISHED" if complete else "NOT_ESTABLISHED",
            "target_cases": target_cases,
            "target_operators": TARGET_OPERATORS,
            **{
                key: value if complete else None
                for key, value in subset.items()
                if key.endswith("percent") or key.startswith("sum_")
            },
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "operator-comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(output_dir / "operator-summary.json", summary)
    (output_dir / "operator-summary.md").write_text(
        _markdown(summary), encoding="utf-8"
    )

    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory containing plan/environment/results JSON",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Summary artifact directory"
    )
    parser.add_argument(
        "--allow-partial-plan",
        action="store_true",
        help="Audit an explicitly selected smoke subset without establishing full-target metrics.",
    )
    args = parser.parse_args(argv)
    summary = aggregate(
        args.input, args.output, require_full_plan=not args.allow_partial_plan
    )
    print(
        json.dumps(
            {
                "audit_passed": summary["audit_passed"],
                "valid_cases": summary["valid_subset"]["valid_cases"],
                "represented_operators": summary["valid_subset"][
                    "operators_with_valid_pairs"
                ],
                "complete_operators": summary["valid_subset"][
                    "fully_covered_operators"
                ],
                "formal_status": summary["formal_full_matrix"]["status"],
                "audit_error_count": len(summary["audit_errors"]),
            }
        )
    )

    return 0 if summary["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
