import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from benchmarks.ascend_report import case_profiles as profiles


def make_case(operator, label="small", **values):
    return {
        "id": f"{operator}__bfloat16__{label}",
        "operator": operator,
        "label": label,
        "dtype": "bfloat16",
        "layout": "contiguous",
        "shape": [4, 8],
        **values,
    }


@pytest.fixture
def original_plan():
    return [
        make_case("clamp"),
        make_case("dropout"),
        make_case("dropout", "p_zero", parameters={"p": 0.0}),
        make_case("dropout", "p_one", parameters={"p": 1.0}),
        make_case("dropout", "inference", parameters={"p": 0.9, "training": False}),
        make_case(
            "max_pool2d",
            shape=[1, 3, 16, 16],
            parameters={
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "return_indices": True,
            },
        ),
        make_case("reduction", "prod", parameters={"kind": "prod", "dim": -1}),
        make_case("reduction", "sum", parameters={"kind": "sum", "dim": -1}),
        make_case("addmm", parameters={"alpha": 0.75, "beta": -0.5}),
        make_case("addmm", "default"),
        make_case(
            "conv2d",
            "grouped_dilated",
            shape=[2, 6, 31, 33],
            weight_shape=[10, 3, 3, 2],
            parameters={
                "stride": [2, 1],
                "padding": [1, 1],
                "dilation": [1, 2],
                "groups": 2,
            },
        ),
        make_case("abs"),
    ]


def test_csv_default_keeps_original_cases_and_nested_values(original_plan):
    selected = profiles.apply_profile(original_plan)
    assert selected == original_plan
    selected[0]["shape"][0] = 999
    selected[5]["parameters"]["return_indices"] = False
    assert original_plan[0]["shape"] == [4, 8]
    assert original_plan[5]["parameters"]["return_indices"] is True


def test_common_profile_has_distinct_ids_and_original_definitions(original_plan):
    before = copy.deepcopy(original_plan)
    selected = profiles.apply_profile(original_plan, profiles.COMMON_CONTRACTS_PROFILE)
    assert original_plan == before
    assert len(selected) == 10
    assert len({case["id"] for case in selected}) == 10
    originals = {case["id"]: case for case in original_plan}
    assert selected[-1] == original_plan[-1]
    for case in selected[:-1]:
        original = originals[case["original_id"]]
        assert case["id"] == original["id"] + "__common_v1"
        assert case["case_profile"] == "common_contracts_v1"
        assert case["original_case"] == original
        assert (
            case["original_case_sha256"]
            == hashlib.sha256(
                json.dumps(
                    original, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            ).hexdigest()
        )
    selected[0]["original_case"]["shape"][0] = 999
    assert original_plan == before
    with pytest.raises(ValueError, match="original plan"):
        profiles.apply_profile(selected, profiles.COMMON_CONTRACTS_PROFILE)


def test_common_profile_changes_only_the_selected_contracts(original_plan):
    selected = profiles.apply_profile(original_plan, profiles.COMMON_CONTRACTS_PROFILE)
    cases = {(case["operator"], case["label"]): case for case in selected}
    assert ("dropout", "p_one") not in cases
    assert ("reduction", "prod") not in cases
    assert cases["clamp", "small"]["parameters"] == {"min": -1.0, "max": 1.0}
    assert cases["dropout", "small"]["parameters"] == {
        "p": 0.25,
        "training": False,
        "return_mask": False,
    }
    assert cases["dropout", "p_zero"]["parameters"] == {
        "p": 0.0,
        "training": True,
        "return_mask": False,
    }
    assert cases["dropout", "inference"]["parameters"] == {
        "p": 0.9,
        "training": False,
        "return_mask": False,
    }
    assert cases["max_pool2d", "small"]["parameters"] == {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "return_indices": False,
    }
    assert cases["reduction", "sum"]["parameters"] == {"kind": "sum", "dim": -1}
    for label in ("small", "default"):
        assert cases["addmm", label]["parameters"] == {"alpha": 1.0, "beta": 1.0}
    conv = cases["conv2d", "grouped_dilated"]
    assert conv["weight_shape"] == [10, 6, 3, 2]
    assert conv["parameters"] == {
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 1,
    }


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def clamp(self, *, min, max):
        return FakeTensor(
            [min if x < min else max if x > max else x for x in self.values]
        )


@pytest.fixture
def baseline(monkeypatch):
    state = SimpleNamespace(calls=[], preparations=[], output=object())

    def get_operator(operator):
        def call(*args, **parameters):
            state.calls.append((operator, args, parameters))
            return state.output

        return call

    state.module = SimpleNamespace(get_operator=get_operator)
    monkeypatch.setitem(sys.modules, "triton_ascend_ops", state.module)

    def prepare(case, device):
        state.preparations.append((case, device))
        x = FakeTensor([-2.0, -0.75, 0.5, 2.0])
        operator = case["operator"]
        parameters = (
            {"min": -0.25, "max": 0.75}
            if operator == "clamp"
            else {
                "p": case.get("parameters", {}).get("p", 0.25),
                "training": case.get("parameters", {}).get("training", True),
                "seed": 123,
                "offset": 17,
                "return_mask": True,
            }
            if operator == "dropout"
            else dict(case.get("parameters", {}))
        )
        state.prepared = {
            "args": [x],
            "parameters": parameters,
            "call": lambda: get_operator(operator)(x, **parameters),
            "reference": x.clamp(**parameters) if operator == "clamp" else x,
            "exact": False,
            "flops": None,
            "logical_bytes": 16,
        }
        return state.prepared

    state.prepare = prepare
    return state


@pytest.mark.parametrize("operator", ["clamp", "dropout"])
def test_prepare_overrides_hardcoded_kwargs_and_uses_baseline_call(baseline, operator):
    case = profiles.apply_profile(
        [make_case(operator)], profiles.COMMON_CONTRACTS_PROFILE
    )[0]
    prepared = profiles.prepare(case, "cpu", baseline.prepare)
    assert len(baseline.preparations) == 1
    assert not baseline.calls
    assert prepared["parameters"] == case["parameters"]
    assert prepared["call"]() is baseline.output
    assert baseline.calls == [(operator, tuple(prepared["args"]), case["parameters"])]
    assert baseline.module.get_operator is sys.modules["triton_ascend_ops"].get_operator
    assert "case_profile_metadata" not in baseline.prepared
    if operator == "clamp":
        assert prepared["reference"].values == [-1.0, -0.75, 0.5, 1.0]
        assert baseline.prepared["reference"].values == [-0.25, -0.25, 0.5, 0.75]
    else:
        assert prepared["reference"] is baseline.prepared["reference"]
        assert baseline.prepared["parameters"]["return_mask"] is True
        assert prepared["case_profile_metadata"]["random_generation"] is False
        assert prepared["case_profile_metadata"][
            "baseline_internal_mask_writes_retained"
        ]


def test_pool_prepare_retains_original_call_and_internal_indices_note(baseline):
    case = profiles.apply_profile(
        [make_case("max_pool2d", parameters={"kernel_size": 3})],
        profiles.COMMON_CONTRACTS_PROFILE,
    )[0]
    prepared = profiles.prepare(case, "cpu", baseline.prepare)
    assert prepared["call"] is baseline.prepared["call"]
    assert prepared["reference"] is baseline.prepared["reference"]
    assert prepared["parameters"] == case["parameters"]
    assert prepared["case_profile_metadata"][
        "baseline_internal_indices_writes_retained"
    ]


@pytest.mark.parametrize("operator", ["dropout", "max_pool2d", "clamp", "addmm"])
def test_verify_uses_existing_tolerances_and_does_not_mutate_case(operator):
    case = profiles.apply_profile(
        [make_case(operator)], profiles.COMMON_CONTRACTS_PROFILE
    )[0]
    before = copy.deepcopy(case)
    prepared, output, checks = object(), object(), object()
    seen = []

    def original_verify(current, actual_prepared, actual_output, phase):
        seen.append((current, actual_prepared, actual_output, phase))
        return checks

    assert profiles.verify(case, prepared, output, "after", original_verify) is checks
    current, actual_prepared, actual_output, phase = seen[0]
    assert actual_prepared is prepared and actual_output is output and phase == "after"
    assert current["operator"] == (
        "abs" if operator in {"dropout", "max_pool2d"} else operator
    )
    assert current["dtype"] == case["dtype"]
    assert case == before


def test_unprofiled_prepare_and_verify_keep_original_contract(baseline):
    case = make_case("dropout")
    prepared = profiles.prepare(case, "cpu", baseline.prepare)
    assert prepared is baseline.prepared
    seen = []
    profiles.verify(case, prepared, None, "before", lambda *args: seen.append(args))
    assert seen[0][0] is case
    assert prepared["parameters"]["return_mask"] is True


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown case profile"):
        profiles.apply_profile([], "unknown")
    with pytest.raises(ValueError, match="Unknown case profile"):
        profiles.prepare(
            make_case("dropout", case_profile="unknown"), "cpu", lambda *_: None
        )


@pytest.mark.parametrize(
    "changes",
    [{"training": True}, {"return_mask": True}, {"seed": 1}, {"p": float("nan")}],
)
def test_identity_profile_rejects_stochastic_or_mask_contracts(baseline, changes):
    case = profiles.apply_profile(
        [make_case("dropout")], profiles.COMMON_CONTRACTS_PROFILE
    )[0]
    case["parameters"].update(changes)
    with pytest.raises(ValueError, match="values-only identity"):
        profiles.prepare(case, "cpu", baseline.prepare)
    assert not baseline.calls
