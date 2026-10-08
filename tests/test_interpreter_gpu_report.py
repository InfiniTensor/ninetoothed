"""Exercise the report CLI without pretending its test doubles validate a GPU."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_interpreter_gpu.py"


def invoke(monkeypatch, path, backend):
    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", [str(SCRIPT), "--report", str(path)])
        patch.setattr(sys, "path", sys.path.copy())
        patch.setitem(sys.modules, "tests.test_interpreter_gpu", backend)

        with pytest.raises(SystemExit) as stopped:
            runpy.run_path(str(SCRIPT), run_name="__main__")
    return stopped.value.code


@pytest.fixture
def backend():
    cases = [
        SimpleNamespace(name=f"case_{i}", category="test_double") for i in range(3)
    ]
    torch = SimpleNamespace(
        __version__="test_double",
        version=SimpleNamespace(cuda="test_double"),
        cuda=SimpleNamespace(
            get_device_name=lambda device: "test_double",
            get_device_capability=lambda device: (0, 0),
        ),
    )

    return SimpleNamespace(
        GPU_CASES=cases,
        SEED=2026,
        require_gpu=lambda device: (torch, SimpleNamespace(__version__="test_double")),
        run_gpu_case=lambda case, torch, device: {
            "name": case.name,
            "program": case.name,
            "category": case.category,
            "status": "PASS",
        },
    )


@pytest.mark.parametrize("kind", ["file", "symlink", "broken_symlink", "directory"])
def test_existing_destination_is_never_overwritten(
    monkeypatch, tmp_path, backend, kind
):
    path, target = tmp_path / "report.json", tmp_path / "prior.json"
    evidence = b'{"status":"PASS","marker":"preserve prior evidence"}\n'

    if kind == "file":
        path.write_bytes(evidence)
    elif kind == "directory":
        path.mkdir()
    else:
        if kind == "symlink":
            target.write_bytes(evidence)

        path.symlink_to(target)

    def forbidden(*args):
        pytest.fail("GPU work started despite an occupied report path.")

    backend.require_gpu = forbidden
    assert invoke(monkeypatch, path, backend) == 2

    if kind == "file":
        assert path.read_bytes() == evidence
    elif kind == "directory":
        assert path.is_dir()
    else:
        assert path.is_symlink()
        assert (
            target.read_bytes() == evidence
            if kind == "symlink"
            else not target.exists()
        )


def test_unavailable_gpu_is_recorded_as_unverified(monkeypatch, tmp_path, backend):
    def unavailable(device):
        raise RuntimeError("GPU unavailable test double.")

    backend.require_gpu = unavailable
    path = tmp_path / "nested/report.json"
    assert invoke(monkeypatch, path, backend) == 2
    report = json.loads(path.read_text())
    assert report["status"] == "UNVERIFIED"
    assert report["cases"] == []
    assert "GPU unavailable test double" in report["error"]


def test_completed_report_contains_all_cases(monkeypatch, tmp_path, backend):
    path = tmp_path / "report.json"
    assert invoke(monkeypatch, path, backend) == 0
    report = json.loads(path.read_text())
    assert report["status"] == "PASS"
    assert report["passed_cases"] == report["total_cases"] == 3
    assert report["passed_programs"] == [case.name for case in backend.GPU_CASES]


def test_case_failure_does_not_hide_later_results(monkeypatch, tmp_path, backend):
    original = backend.run_gpu_case

    def failing(case, torch, device):
        if case.name == "case_1":
            raise AssertionError("Incorrect output.")
        return original(case, torch, device)

    backend.run_gpu_case = failing
    path = tmp_path / "report.json"
    assert invoke(monkeypatch, path, backend) == 1
    report = json.loads(path.read_text())
    assert report["status"] == "FAIL"
    assert [case["status"] for case in report["cases"]] == ["PASS", "FAIL", "PASS"]
    assert report["passed_cases"] == 2


def test_interrupt_preserves_completed_cases(monkeypatch, tmp_path, backend):
    original = backend.run_gpu_case
    path = tmp_path / "report.json"

    def interrupt(case, torch, device):
        if case.name == "case_1":
            raise KeyboardInterrupt
        return original(case, torch, device)

    backend.run_gpu_case = interrupt
    assert invoke(monkeypatch, path, backend) == 130
    report = json.loads(path.read_text())
    assert report["status"] == "INTERRUPTED"
    assert [case["name"] for case in report["cases"]] == ["case_0"]
    assert report["active_case"] == "case_1"
    assert report["passed_cases"] == 1
    assert report["total_cases"] == 3


def test_progress_is_readable_before_next_case(monkeypatch, tmp_path, backend):
    original = backend.run_gpu_case
    path = tmp_path / "report.json"

    def observe(case, torch, device):
        report = json.loads(path.read_text())
        assert report["status"] == "RUNNING"
        assert report["active_case"] == case.name
        assert len(report["cases"]) == int(case.name[-1])

        return original(case, torch, device)

    backend.run_gpu_case = observe
    assert invoke(monkeypatch, path, backend) == 0
