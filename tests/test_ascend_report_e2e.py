import copy
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.ascend_report import affinity, e2e
from benchmarks.ascend_report import e2e_runner as runner


@pytest.fixture
def configuration(tmp_path):
    raw = {
        "paths": {
            "ninetoothed": "compiler",
            "ntops": "operators",
            "baseline": "baseline",
            "model": "model",
        },
        "output_dir": "output",
        "cache_dir": "cache",
        "tmp_dir": "temporary",
        "device": 0,
        "e2e": {
            "prompt": "Synthetic test prompt.",
            "model_revision": "a" * 40,
            "stages": [1, 2, 3, 4],
        },
    }
    path = tmp_path / "config.json"
    runner.write_json(path, raw)
    config = runner.read_config(path)

    return SimpleNamespace(
        root=tmp_path,
        path=path,
        raw=raw,
        config=config,
        output=Path(config["e2e_output_dir"]),
    )


@pytest.fixture
def harness(configuration, monkeypatch):
    state = configuration
    state.sources = {"sha256": {"compiler.py": "source-a"}, "commits": {}}
    state.model = {"files": {"weights": {"sha256": "weights-a", "bytes": 3}}}
    state.invocations = []
    state.device_invocations = []
    state.active_workers = {}
    state.maximum_concurrency = 0
    state.reference_barrier = None
    state.corrupt_uuid = False
    state.worker_lock = threading.Lock()

    def worker_record(backend, stage, order, identity, device=0, cpus=None):
        output = {
            "output_ids": [101 if backend != "ntops" else 102] * 128,
            "output_text": "Synthetic alpha output."
            if backend != "ntops"
            else "Synthetic beta output.",
            "sanity": {"passed": True},
        }
        record = {
            "status": "PASS",
            "run_identity": identity,
            "backend": backend,
            "stage": stage,
            "order": order,
            "model": {"id": "synthetic", "input_tokens": 19, "output_tokens": 128},
            "environment": {
                "versions": {},
                "physical_device": device,
                "device_uuid": f"synthetic-device-{device}",
                "cpu_affinity": cpus,
            },
            **output,
        }

        if backend != "reference":
            seconds = 1.0 if backend == "triton" else 2.0
            record.update(
                samples_seconds=[seconds] * 10,
                median_seconds=seconds,
                sample_outputs=[copy.deepcopy(output) for _ in range(10)],
                operator_calls={"rms_norm": 1, "silu": 1},
                semantic_review={"status": "pending"},
            )

        return record

    def run_process(command, log, timeout):
        def argument(name):
            return command[command.index(name) + 1]

        backend = argument("--backend")
        stage = int(argument("--stage"))
        order = argument("--order")
        device = int(argument("--device"))
        cpus = (
            list(map(int, argument("--cpu-affinity").split(",")))
            if "--cpu-affinity" in command
            else None
        )

        with state.worker_lock:
            assert state.active_workers.get(device, 0) == 0
            state.active_workers[device] = 1
            state.maximum_concurrency = max(
                state.maximum_concurrency, sum(state.active_workers.values())
            )
            state.invocations.append((order, stage, backend))
            state.device_invocations.append((order, stage, backend, device))

        try:
            if backend == "reference" and state.reference_barrier is not None:
                state.reference_barrier.wait(timeout=10)

            record = worker_record(
                backend, stage, order, argument("--identity"), device, cpus
            )

            if state.corrupt_uuid and backend == "ntops":
                record["environment"]["device_uuid"] = "different-physical-device"

            runner.write_json(
                e2e.record_path(state.output, order, stage, backend, device), record
            )
        finally:
            with state.worker_lock:
                state.active_workers[device] = 0

        return 0

    def invoke(*arguments):
        monkeypatch.setattr(
            sys, "argv", ["e2e.py", "--config", str(state.path), *arguments]
        )

        return e2e.main()

    def report():
        return json.loads(
            (state.output / "report-data.json").read_text(encoding="utf-8")
        )

    monkeypatch.setitem(
        sys.modules,
        "fcntl",
        SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *_: None),
    )
    monkeypatch.setattr(e2e, "source_identity", lambda _: state.sources)
    monkeypatch.setattr(e2e, "model_identity", lambda _: state.model)
    monkeypatch.setattr(e2e, "run_process", run_process)
    state.invoke = invoke
    state.report = report
    state.worker_record = worker_record

    return state


def test_config_resolves_paths_and_preserves_protocol(configuration):
    state = configuration
    assert state.config["paths"]["ntops"] == str(state.root / "operators")
    assert state.output == state.root / "output" / "e2e"
    assert [
        state.config["e2e"][name]
        for name in ("input_tokens", "output_tokens", "warmup", "samples")
    ] == [19, 128, 2, 10]
    del state.raw["e2e"]["stages"]
    runner.write_json(state.path, state.raw)
    assert runner.read_config(state.path)["e2e"]["stages"] == [1, 2, 3, 4, 5, 6]
    state.raw["devices"] = [0, 1, 2, 3]
    state.raw["e2e"]["devices"] = [2, 3]
    runner.write_json(state.path, state.raw)
    configured = runner.read_config(state.path)
    assert configured["devices"] == [0, 1, 2, 3]
    assert configured["e2e"]["devices"] == [2, 3]
    assert configured["device"] == 2
    state.raw["scheduling"] = {"TRITON_ALL_BLOCKS_PARALLEL": "1"}
    runner.write_json(state.path, state.raw)

    with pytest.raises(ValueError, match="PARALLEL=0"):
        runner.read_config(state.path)


def test_text_sanity_accepts_unicode_without_judging_semantics():
    assert runner.text_sanity([1] * 128, "这是正常的测试文本。\n第二行。", 128)[
        "passed"
    ]
    assert not runner.text_sanity([1] * 128, "bad\ufffd", 128)["passed"]
    assert not runner.text_sanity([1] * 128, "bad\x00", 128)["passed"]
    assert not runner.text_sanity([1], "Synthetic text.", 128)["passed"]


@pytest.mark.parametrize("stages", [[], [0], [7], [1, 1], [True], "5,6"])
def test_rejects_invalid_stage_selection(configuration, stages):
    configuration.raw["e2e"]["stages"] = stages
    runner.write_json(configuration.path, configuration.raw)
    with pytest.raises(ValueError, match="stages"):
        runner.read_config(configuration.path)


def test_core_coverage_requires_real_pinned_kernel_launches():
    calls = {"sdpa": 32, "linear:q_proj": 32, "linear:lm_head": 1}
    launches = {
        "sdpa:generated:ntops_lab_flash_attention_ascend_npu": 32,
        "linear_mm:generated:ntops_lab_mm_npu": 33,
    }
    coverage = runner.validate_core_launches("ntops", 6, calls, launches)
    assert (
        coverage["sdpa"]["verified"] and coverage["linear_mm"]["kernel_launches"] == 33
    )
    with pytest.raises(RuntimeError, match="real JIT launches"):
        runner.validate_core_launches("ntops", 6, calls, {})
    with pytest.raises(RuntimeError, match="unexpected kernel"):
        runner.validate_core_launches(
            "ntops", 5, calls, {"sdpa:native:torch_fallback": 32}
        )


def test_core_operator_stages_use_selected_devices_and_both_orders(harness):
    harness.raw["devices"] = [0, 1, 2, 3]
    harness.raw["e2e"]["stages"] = [4, 5, 6]
    runner.write_json(harness.path, harness.raw)
    assert harness.invoke() == 0
    assert len(harness.invocations) == 3 + 3 * 4
    assert [stage["stage"] for stage in harness.report()["stages"]] == [4, 5, 6]
    for stage, device in zip([4, 5, 6], [0, 1, 2]):
        selected = [item for item in harness.device_invocations if item[1] == stage]
        assert len(selected) == 4
        assert {item[3] for item in selected} == {device}
        assert [item[2] for item in selected] == ["triton", "ntops", "ntops", "triton"]


def test_two_orders_use_seventeen_workers_and_resume_reuses_results(harness):
    assert harness.invoke() == 0
    assert len(harness.invocations) == 17
    assert harness.invocations[0] == ("reference", 0, "reference")

    for stage in range(1, 5):
        forward = [
            item[2]
            for item in harness.invocations
            if item[:2] == ("triton_then_ntops", stage)
        ]
        reverse = [
            item[2]
            for item in harness.invocations
            if item[:2] == ("ntops_then_triton", stage)
        ]
        assert forward == ["triton", "ntops"]
        assert reverse == ["ntops", "triton"]

    for stage in harness.report()["stages"]:
        assert not stage["accuracy"]["tokens_equal"]
        assert not stage["accuracy"]["text_equal"]
        assert stage["accuracy"]["passed"] is None
        assert stage["timings"]["forward"]["ratio_percent"] == 50.0
        assert stage["timings"]["forward"]["acceptance_eligible"] is None

    assert harness.invoke("--resume") == 0
    assert len(harness.invocations) == 17


def test_four_device_lanes_run_concurrently_with_fixed_stage_pairing(
    harness, monkeypatch
):
    harness.raw["devices"] = [0, 1, 2, 3]
    runner.write_json(harness.path, harness.raw)
    monkeypatch.setattr(
        affinity.os, "sched_getaffinity", lambda _: set(range(8)), raising=False
    )
    monkeypatch.setattr(
        affinity.os, "sched_setaffinity", lambda *_: None, raising=False
    )
    harness.reference_barrier = threading.Barrier(4)
    assert harness.invoke() == 0
    assert len(harness.invocations) == 20
    assert harness.maximum_concurrency == 4

    for device in range(4):
        lane = [item for item in harness.device_invocations if item[3] == device]
        assert lane[0] == ("reference", 0, "reference", device)
        assert [(item[0], item[2]) for item in lane[1:]] == [
            ("triton_then_ntops", "triton"),
            ("triton_then_ntops", "ntops"),
            ("ntops_then_triton", "ntops"),
            ("ntops_then_triton", "triton"),
        ]
        assert {item[1] for item in lane[1:]} == {device + 1}

    report = harness.report()
    assert set(report["reference_by_device"]) == {"0", "1", "2", "3"}
    assert report["provenance"]["execution"]["max_concurrent_workers"] == 4
    assert report["provenance"]["observed_max_concurrent_workers"] == 4

    for stage in report["stages"]:
        device = stage["stage"] - 1
        assert stage["physical_device"] == device
        assert (
            stage["timings"]["forward"]["device_uuid"] == f"synthetic-device-{device}"
        )
        assert stage["timings"]["reverse"]["cpu_affinity"] == [
            2 * device,
            2 * device + 1,
        ]

    assert harness.invoke("--resume") == 0
    assert len(harness.invocations) == 20


def test_pairing_rejects_a_different_physical_uuid(harness):
    harness.corrupt_uuid = True

    with pytest.raises(ValueError, match="different physical UUIDs"):
        harness.invoke()


def test_cpu_affinity_reserves_explicit_sets_and_binds_before_runtime(monkeypatch):
    current = set(range(8))

    def apply(pid, selected):
        current.clear()
        current.update(selected)

    monkeypatch.setattr(
        affinity.os, "sched_getaffinity", lambda _: current.copy(), raising=False
    )
    monkeypatch.setattr(affinity.os, "sched_setaffinity", apply, raising=False)
    planned = affinity.plan_cpu_affinity([0, 1, 2], {"1": [2, 3]}, minimum=1)
    assert planned == {0: [0, 1, 4], 1: [2, 3], 2: [5, 6, 7]}

    with pytest.raises(ValueError, match="overlap"):
        affinity.plan_cpu_affinity([0, 1], {"0": [0], "1": [0]})

    with pytest.raises(ValueError, match="outside"):
        affinity.plan_cpu_affinity([0], {"0": [99]})

    with pytest.raises(ValueError, match="Not enough"):
        affinity.plan_cpu_affinity([0, 1, 2], minimum=3)

    assert affinity.apply_cpu_affinity(planned[1]) == [2, 3]
    assert affinity.apply_cpu_affinity(None) == [2, 3]


def test_resume_rejects_changed_sources_model_and_raw_results(harness):
    harness.invoke()
    harness.sources["sha256"]["compiler.py"] = "changed"

    with pytest.raises(ValueError, match="changed configuration, model or source"):
        harness.invoke("--resume")

    harness.sources["sha256"]["compiler.py"] = "source-a"
    harness.model["files"]["weights"]["sha256"] = "changed"

    with pytest.raises(ValueError, match="changed configuration, model or source"):
        harness.invoke("--resume")

    harness.model["files"]["weights"]["sha256"] = "weights-a"
    path = harness.output / "results" / "triton_then_ntops_stage1_triton.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["output_text"] = "Edited text."
    runner.write_json(path, record)

    with pytest.raises(ValueError, match="raw result changed"):
        harness.invoke("--resume")


def test_text_review_is_hash_bound_and_does_not_require_equal_tokens(harness):
    harness.invoke()
    reviews = {
        path.relative_to(harness.output).as_posix(): {
            "raw_sha256": runner.sha256_file(path),
            "passed": True,
            "reviewer": "Synthetic test reviewer",
        }
        for path in (harness.output / "results").glob("*.json")
    }
    runner.write_json(harness.output / "semantic-review.json", {"records": reviews})
    assert harness.invoke("--summarize-only") == 0
    report = harness.report()
    assert all(stage["accuracy"]["passed"] for stage in report["stages"])
    assert not report["stages"][0]["accuracy"]["tokens_equal"]
    assert report["stages"][0]["timings"]["forward"]["acceptance_eligible"]
    next(iter(reviews.values()))["raw_sha256"] = "older-record"
    runner.write_json(harness.output / "semantic-review.json", {"records": reviews})

    with pytest.raises(ValueError, match="older raw result"):
        harness.invoke("--summarize-only")


def test_saved_median_must_match_raw_samples(harness):
    record = harness.worker_record("triton", 1, "forward", "identity")
    record["worker_returncode"] = 0
    record["median_seconds"] = 99.0

    with pytest.raises(ValueError, match="median"):
        e2e.validate_record(record, "identity", harness.config["e2e"])


def test_identity_hashes_actual_source_and_model_contents(configuration):
    state = configuration

    for role, suffix in (
        ("compiler", "src/ninetoothed"),
        ("operators", "src/ntops_lab"),
        ("baseline", "triton_ascend_ops"),
    ):
        directory = state.root / role / suffix
        directory.mkdir(parents=True)
        (directory / "entry.py").write_text("value = 1\n", encoding="utf-8")

    model = state.root / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "weights.safetensors").write_bytes(b"123")
    sources = e2e.source_identity(state.config)
    weights = e2e.model_identity(state.config)
    original = e2e.identity_digest(state.config, sources, weights)
    (model / "weights.safetensors").write_bytes(b"124")
    assert original != e2e.identity_digest(
        state.config, sources, e2e.model_identity(state.config)
    )
    (state.root / "compiler/src/ninetoothed/entry.py").write_text(
        "value = 2\n", encoding="utf-8"
    )
    assert original != e2e.identity_digest(
        state.config, e2e.source_identity(state.config), weights
    )


def test_logit_differences_remain_diagnostic(tmp_path):
    records = []

    for name, value in (("triton", 0.0), ("ntops", 1.0)):
        path = tmp_path / (name + ".npz")
        np.savez_compressed(
            path,
            prefill=np.full((1, 4), value, dtype=np.float32),
            decode=np.full((1, 4), value, dtype=np.float32),
        )
        records.append(
            {
                "logits_diagnostic": {
                    "path": str(path),
                    "sha256": runner.sha256_file(path),
                }
            }
        )

    diagnostic = e2e.logits_diagnostics(*records)
    assert diagnostic["prefill"]["mismatch_count"] == 4
    assert not diagnostic["prefill"]["passed"]
    assert not diagnostic["acceptance_gate"]
