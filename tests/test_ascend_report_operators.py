import json

import pytest

from benchmarks.ascend_report import operators


def _record():
    return {
        "id": "add_case",
        "operator": "add",
        "dtype": "float32",
        "label": "strided",
        "layout": "transpose",
        "status": "PASS",
        "inputs": [
            {"shape": [3, 5], "stride": [1, 3]},
            {"shape": [3, 5], "stride": [5, 1]},
        ],
        "parameters": {"alpha": 1},
    }


@pytest.fixture
def csv_case(monkeypatch):
    record = _record()
    expected = {
        key: record[key] for key in ("id", "operator", "dtype", "label", "layout")
    }
    expected.update(
        input_shapes=json.dumps([[3, 5], [3, 5]]),
        input_strides=json.dumps([[1, 3], [5, 1]]),
        parameters=json.dumps({"alpha": 1}),
    )
    monkeypatch.setattr(operators, "CASES_CSV", {record["id"]: expected})
    return record


def test_csv_contract_matches_actual_tensor_metadata(csv_case):
    operators.verify_csv_record(csv_case)


@pytest.mark.parametrize("field", ("shape", "stride"))
def test_csv_contract_rejects_changed_layout(csv_case, field):
    csv_case["inputs"][0][field] = [5, 3]
    with pytest.raises(ValueError, match="CSV mismatch"):
        operators.verify_csv_record(csv_case)


def test_csv_contract_rejects_changed_parameters(csv_case):
    csv_case["parameters"]["alpha"] = 2
    with pytest.raises(ValueError, match="parameters"):
        operators.verify_csv_record(csv_case)


def test_failed_case_does_not_require_measurement_metadata(csv_case):
    csv_case.update(status="FAIL")
    del csv_case["inputs"]
    operators.verify_csv_record(csv_case)
