import csv
import json

import pytest

from benchmarks.ascend_report.summarize import aggregate

_PHASES = (
    "public_before_timing",
    "graph_replay_before_timing",
    "graph_after_timing",
    "public_after_timing",
)


def _check(phase):
    return {
        "phase": phase,
        "name": "output",
        "passed": True,
        "mismatch_count": 0,
        "max_abs_error": 0.0,
        "rtol": 0.0005,
        "atol": 0.0005,
    }


def _stats(value, count):
    return {
        "samples_us": [value] * count,
        "median_us": value,
        "p95_us": value,
        "min_us": value,
        "max_us": value,
        "mean_us": value,
        "cv_pct": 0.0,
    }


def _record(case, implementation, elapsed):
    arity = 2 if case["operator"] == "add" else 1

    return {
        **case,
        "implementation": implementation,
        "status": "PASS",
        "inputs": [
            {
                "shape": [4],
                "stride": [1],
                "dtype": "torch.float32",
                "device": "npu:0",
            }
            for _ in range(arity)
        ],
        "parameters": {},
        "checks": [_check(phase) for phase in _PHASES],
        "preparation": {"checks": [_check("precompile_validation")]},
        "prepared_device": _stats(elapsed, 3),
        "public_api": _stats(elapsed * 10, 4),
        "device_samples": 3,
        "api_samples": 4,
        "warmup_per_path": 2,
        "runtime": {
            "physical_device": 0,
            "device_properties": "SyntheticDevice(uuid=synthetic-card-zero)",
            "versions": {
                "torch": "synthetic-version",
                "torch_npu": "synthetic-version",
                "triton-ascend": "synthetic-version",
            },
            "cann": "synthetic-version",
            "triton_import": "synthetic-version",
            "python": "synthetic-version",
        },
        "kernels": [{"name": "synthetic_kernel", "compiled_type": "CompiledKernel"}],
        "launches_per_call": 1,
        "graph_nodes": 8,
        "graph_repeats_per_sample": 2,
        "adapter": {
            "mode": "adapted",
            "implementation_type": "ninetoothed_generated",
            "expected_triton_launches": 1,
            "output_contract": "tensor",
        },
    }


@pytest.fixture
def experiment(tmp_path):
    plan = [
        {
            "id": f"{operator}__synthetic_{index}",
            "operator": operator,
            "dtype": "float32",
            "label": f"synthetic_{index}",
            "layout": "contiguous",
            "shape": [4],
        }
        for index, operator in enumerate(("abs", "abs", "add"))
    ]
    elapsed = ((4.0, 2.0), (5.0, 10.0), (8.0, 2.0))
    records = [
        _record(case, implementation, value)
        for case, values in zip(plan, elapsed)
        for implementation, value in zip(("baseline", "ninetoothed_adapted"), values)
    ]
    environment = {
        "case_count": 3,
        "operator_count": 2,
        "measurement": {"warmup": 2, "device_samples": 3, "api_samples": 4},
        "assignment": {"0": ["abs", "add"]},
        "source_unchanged_at_end": True,
    }
    source = tmp_path / "operators"
    source.mkdir()
    for name, value in (("plan.json", plan), ("environment.json", environment)):
        (source / name).write_text(json.dumps(value), encoding="utf-8")

    def run(values=records, *, require_full_plan=True):
        (source / "results.json").write_text(json.dumps(values), encoding="utf-8")
        output = tmp_path / "summary"
        summary = aggregate(source, output, require_full_plan=require_full_plan)
        with (output / "operator-comparison.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))

        return summary, rows

    return records, run


def test_valid_pairs_keep_distinct_weightings_and_partial_formal_status(experiment):
    _, run = experiment
    summary, rows = run()
    subset = summary["valid_subset"]

    assert subset["valid_cases"] == 3
    assert subset["operators_with_valid_pairs"] == 2
    assert subset["case_geomean_percent"] == pytest.approx(100 * 4 ** (1 / 3))
    assert subset["operator_equal_geomean_percent"] == pytest.approx(200)
    assert subset["case_arithmetic_mean_ratio_percent"] == pytest.approx(100 * 6.5 / 3)
    assert subset["sum_baseline_us"] == 17
    assert subset["sum_ninetoothed_us"] == 14
    assert subset["ratio_of_summed_times_percent"] == pytest.approx(100 * 17 / 14)
    assert summary["formal_full_matrix"]["status"] == "NOT_ESTABLISHED"
    assert summary["formal_full_matrix"]["case_geomean_percent"] is None
    assert summary["formal_full_matrix"]["operator_equal_geomean_percent"] is None
    assert summary["coverage"]["target_cases"] == 465
    assert summary["coverage"]["target_operators"] == 41
    assert len(rows) == 3
    assert all(row["baseline_actual_triton_graph_launches"] == "8" for row in rows)


def test_missing_path_is_retained_but_excluded_from_pairs(experiment):
    records, run = experiment
    missing = records.pop(1)
    summary, rows = run()

    assert summary["valid_subset"]["valid_cases"] == 2
    assert summary["coverage"]["missing_paths"] == [
        (missing["id"], missing["implementation"])
    ]
    assert rows[0]["ninetoothed_status"] == "MISSING"
    assert rows[0]["performance_percent"] == ""
    assert not summary["audit_passed"]


def test_explicit_smoke_subset_is_audited_without_full_target_metrics(experiment):
    _, run = experiment
    summary, _ = run(require_full_plan=False)

    assert summary["audit_passed"]
    assert summary["coverage"]["all_selected_paths_present_once"]
    assert not summary["coverage"]["plan_covers_full_target"]
    assert summary["formal_full_matrix"]["status"] == "NOT_ESTABLISHED"


def test_pair_cpu_affinity_must_match(experiment):
    records, run = experiment
    records[0]["runtime"]["cpu_affinity"] = [0, 1]
    records[1]["runtime"]["cpu_affinity"] = [2, 3]
    summary, _ = run()

    assert summary["valid_subset"]["valid_cases"] == 2
    assert any("cpu_affinity" in error for error in summary["audit_errors"])


def test_duplicate_path_invalidates_only_its_pair(experiment):
    records, run = experiment
    records.append(records[1].copy())
    summary, rows = run()

    assert summary["valid_subset"]["valid_cases"] == 2
    assert summary["coverage"]["duplicate_paths"] == [
        [records[1]["id"], "ninetoothed_adapted"]
    ]
    assert rows[0]["performance_percent"] == ""
    assert "Duplicate measurement path" in rows[0]["ninetoothed_audit_errors"]


def test_saved_median_must_match_raw_samples(experiment):
    records, run = experiment
    records[1]["prepared_device"]["median_us"] = 200
    summary, rows = run()

    assert summary["valid_subset"]["valid_cases"] == 2
    assert "prepared_device.median_us" in rows[0]["ninetoothed_audit_errors"]
    assert rows[0]["performance_percent"] == ""


def test_nonfinite_sample_invalidates_pair(experiment):
    records, run = experiment
    records[1]["public_api"]["samples_us"][0] = float("nan")
    summary, rows = run()

    assert summary["valid_subset"]["valid_cases"] == 2
    assert "finite" in rows[0]["ninetoothed_audit_errors"]
    assert rows[0]["api_performance_percent"] == ""


def test_same_logical_card_with_different_uuid_is_not_a_pair(experiment):
    records, run = experiment
    records[1]["runtime"]["device_properties"] = "SyntheticDevice(uuid=other-card)"
    summary, rows = run()

    assert summary["valid_subset"]["valid_cases"] == 2
    assert "UUIDs differ" in rows[0]["pair_audit_errors"]
    assert rows[0]["performance_percent"] == ""


def test_relaxed_original_tolerance_is_rejected(experiment):
    records, run = experiment
    records[1]["checks"][0]["rtol"] = 0.5
    summary, rows = run()

    assert summary["valid_subset"]["valid_cases"] == 2
    assert "tolerances differ" in rows[0]["ninetoothed_audit_errors"]


def test_failed_and_unsupported_paths_do_not_enter_performance_means(experiment):
    records, run = experiment
    records[1].update(status="FAIL", error="Synthetic compiler failure")
    records[3].update(status="UNSUPPORTED", error="Synthetic unsupported contract")
    summary, rows = run()

    assert summary["valid_subset"]["valid_cases"] == 1
    assert summary["valid_subset"]["case_geomean_percent"] == pytest.approx(400)
    assert rows[0]["performance_percent"] == rows[1]["performance_percent"] == ""
    counts = summary["measurement_status_counts"]["ninetoothed_adapted"]
    assert counts == {"FAIL": 1, "UNSUPPORTED": 1, "PASS": 1}
