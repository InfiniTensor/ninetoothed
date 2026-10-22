"""Synthetic CPU records exercise provenance and target separation, not speed."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.ascend_report import case_profiles, operators, summarize

_SPEC = importlib.util.spec_from_file_location(
    "ascend_statistics_test_helpers",
    Path(__file__).with_name("test_ascend_report_statistics.py"),
)
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)


def _case(operator="clamp", label="synthetic"):
    case = {
        "id": f"{operator}__float32__{label}",
        "operator": operator,
        "dtype": "float32",
        "label": label,
        "layout": "contiguous",
        "shape": [4],
    }
    if operator.startswith("bitwise"):
        case["dtype"] = "int32"
    if operator in {"mm", "addmm", "bmm"}:
        case.update(dims=[2, 2, 2], batch=1)
    if operator in {"layer_norm", "rms_norm", "softmax", "swiglu", "reduction"}:
        case["shape"] = [2, 4]
    if operator == "reduction":
        case["parameters"] = {"kind": "sum", "dim": -1, "keepdim": False}
    if operator == "rotary_position_embedding":
        case.update(shape=[2, 2, 4], parameters={"interleaved": True})
    if operator == "scaled_dot_product_attention":
        case.update(dims=[1, 1, 1, 2, 2, 4], parameters={"is_causal": False})
    if operator == "conv2d":
        case.update(
            shape=[1, 1, 4, 4], weight_shape=[1, 1, 3, 3], parameters={"padding": 1}
        )
    if operator == "max_pool2d":
        case.update(
            shape=[1, 1, 4, 4],
            parameters={
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "return_indices": True,
            },
        )
    return case


def _record(case, implementation="baseline"):
    record = _HELPERS._record(
        copy.deepcopy(case),
        implementation,
        2.0 if implementation == "baseline" else 1.0,
    )
    shapes, strides = summarize._input_contract(case)
    record["inputs"] = [
        {
            "shape": shape,
            "stride": stride,
            "dtype": f"torch.{case['dtype']}",
            "device": "npu:0",
        }
        for shape, stride in zip(shapes, strides)
    ]
    record["parameters"] = summarize._parameters(case)
    exact = case["operator"] in summarize.BOOLEAN_OPS or case["dtype"] == "int32"
    atol = (
        0.003
        if case["operator"] in {"mm", "addmm", "bmm", "conv2d"}
        else 0.001
        if case["operator"] == "scaled_dot_product_attention"
        else 0.0005
    )
    for check in record["checks"] + record["preparation"]["checks"]:
        check.update(rtol=0 if exact else 0.0005, atol=0 if exact else atol)
    return record


def _csv_row(case):
    record = _record(case)
    row = {key: case[key] for key in ("id", "operator", "dtype", "label", "layout")}
    row.update(
        input_shapes=json.dumps([item["shape"] for item in record["inputs"]]),
        input_strides=json.dumps([item["stride"] for item in record["inputs"]]),
        parameters=json.dumps(record["parameters"]),
    )
    return row


@pytest.fixture
def adjusted_csv(monkeypatch):
    original = _case()
    monkeypatch.setattr(operators, "CASES_CSV", {original["id"]: _csv_row(original)})
    adjusted = case_profiles.apply_profile([original], "common_contracts_v1")[0]
    return original, _record(adjusted)


def test_adjusted_record_verifies_current_contract_and_original_csv(adjusted_csv):
    original, adjusted = adjusted_csv
    assert adjusted["parameters"] == {"min": -1.0, "max": 1.0}
    operators.verify_csv_record(adjusted)
    original_record = _record(original)
    assert original_record["parameters"] == {"min": -0.25, "max": 0.75}
    operators.verify_csv_record(original_record)
    original_record["parameters"] = adjusted["parameters"]
    with pytest.raises(ValueError, match="CSV mismatch"):
        operators.verify_csv_record(original_record)


@pytest.mark.parametrize(
    "mutation", ["shape", "stride", "parameters", "definition", "hash"]
)
def test_changed_current_profile_contract_is_rejected(adjusted_csv, mutation):
    _, record = adjusted_csv
    if mutation in {"shape", "stride"}:
        record["inputs"][0][mutation] = [999]
    elif mutation == "parameters":
        record["parameters"]["min"] = -2.0
    elif mutation == "definition":
        record["shape"] = [999]
    else:
        record["original_case_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs|differ"):
        operators.verify_csv_record(record)


def test_unknown_original_id_cannot_link_to_csv(adjusted_csv):
    original, _ = adjusted_csv
    changed = dict(original, id="unknown-original-case")
    adjusted = case_profiles.apply_profile([changed], "common_contracts_v1")[0]
    with pytest.raises((KeyError, ValueError)):
        operators.verify_csv_record(_record(adjusted))


def test_rehashing_tampered_original_case_does_not_bypass_csv(adjusted_csv):
    original, _ = adjusted_csv
    changed = dict(original, shape=[999])
    adjusted = case_profiles.apply_profile([changed], "common_contracts_v1")[0]
    with pytest.raises(ValueError, match="CSV mismatch.*input_shapes"):
        operators.verify_csv_record(_record(adjusted))


def test_adjusted_record_requires_original_case_provenance(adjusted_csv):
    _, record = adjusted_csv
    del record["original_case"]
    with pytest.raises(ValueError, match="original-case provenance"):
        operators.verify_csv_record(record)


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def experiment(tmp_path):
    original = [_case()]
    plan = case_profiles.apply_profile(original, "common_contracts_v1")
    state = SimpleNamespace(root=tmp_path / "operators", original=original, plan=plan)
    state.root.mkdir()
    state.environment = {
        "case_profile": "common_contracts_v1",
        "original_plan_sha256": hashlib.sha256(
            json.dumps(original, sort_keys=True).encode()
        ).hexdigest(),
        "case_count": len(plan),
        "operator_count": 1,
        "measurement": {"warmup": 2, "device_samples": 3, "api_samples": 4},
        "source_unchanged_at_end": True,
    }

    def run(*, write_original=True, require_full_plan=False):
        if write_original:
            _write(state.root / "original-plan.json", state.original)
        _write(state.root / "environment.json", state.environment)
        _write(state.root / "plan.json", state.plan)
        _write(
            state.root / "results.json",
            [
                _record(case, implementation)
                for case in state.plan
                for implementation in summarize.IMPLEMENTATIONS
            ],
        )
        return summarize.aggregate(
            state.root, tmp_path / "summary", require_full_plan=require_full_plan
        )

    state.run = run
    return state


def test_complete_adjusted_matrix_never_establishes_original_csv_scope(experiment):
    names = set(summarize.POINTWISE_OPS) | set(
        "mm addmm bmm layer_norm rms_norm softmax swiglu reduction rotary_position_embedding "
        "scaled_dot_product_attention conv2d max_pool2d dropout".split()
    )
    original = [_case(name) for name in sorted(names)]
    original += [_case("abs", f"extra-{index}") for index in range(412)]
    for index in range(11):
        case = _case("reduction", f"prod-{index}")
        case["parameters"]["kind"] = "prod"
        original.append(case)
    original.append(dict(_case("dropout", "p_one"), parameters={"p": 1.0}))
    assert len(original) == 465
    experiment.original = original
    experiment.plan = case_profiles.apply_profile(original, "common_contracts_v1")
    experiment.environment.update(
        case_count=453,
        operator_count=41,
        original_plan_sha256=hashlib.sha256(
            json.dumps(original, sort_keys=True).encode()
        ).hexdigest(),
    )
    summary = experiment.run(require_full_plan=True)
    assert summary["audit_passed"], summary["audit_errors"]
    assert summary["coverage"]["target_cases"] == 453
    assert summary["coverage"]["target_paths"] == 906
    assert summary["valid_subset"]["valid_cases"] == 453
    assert summary["adjusted_full_matrix"]["status"] == "ESTABLISHED"
    assert summary["adjusted_full_matrix"][
        "operator_equal_geomean_percent"
    ] == pytest.approx(200)
    formal = summary["formal_full_matrix"]
    assert formal["target_cases"] == 465
    assert formal["status"] == "NOT_ESTABLISHED"
    assert all(
        value is None
        for key, value in formal.items()
        if key.endswith("percent") or key.startswith("sum_")
    )


@pytest.mark.parametrize(
    "mutation", ["missing_file", "missing_hash", "wrong_hash", "changed_plan"]
)
def test_adjusted_summary_requires_bound_original_plan(experiment, mutation):
    if mutation == "missing_hash":
        experiment.environment.pop("original_plan_sha256")
    elif mutation == "wrong_hash":
        experiment.environment["original_plan_sha256"] = "0" * 64
    elif mutation == "changed_plan":
        # Keep both current raw records consistent with the tampered plan. The
        # original-plan transformation gate must still catch the contract change.
        experiment.plan[0]["parameters"]["min"] = -2.0
    summary = experiment.run(write_original=mutation != "missing_file")
    assert not summary["audit_passed"]
    assert any(
        message in " ".join(summary["audit_errors"])
        for message in (
            "requires the original plan",
            "provenance digest differs",
            "differs from the declared profile",
        )
    )
    assert summary["adjusted_full_matrix"]["status"] == "NOT_ESTABLISHED"
    assert summary["formal_full_matrix"]["status"] == "NOT_ESTABLISHED"


def test_adjusted_cases_cannot_be_reclassified_as_original_csv(experiment):
    experiment.environment["case_profile"] = "csv"
    summary = experiment.run()
    assert not summary["audit_passed"]
    assert (
        "Adjusted cases cannot establish the original CSV scope"
        in summary["audit_errors"]
    )
    assert summary["coverage"]["target_cases"] == 465
    assert summary["formal_full_matrix"]["status"] == "NOT_ESTABLISHED"
    assert "adjusted_full_matrix" not in summary
