import pytest
import torch

from tests.capabilities import float8
from tests.capabilities.model import CapabilityResult
from tests.capabilities.runner import run_python_probe


@pytest.fixture(autouse=True)
def clear_float8_probe_cache():
    probe_names = (
        "float8_e5m2_cuda",
        "float8_e5m2_mlu",
    )

    for name in probe_names:
        probe = getattr(float8, name, None)

        if probe is not None:
            probe.cache_clear()

    yield

    for name in probe_names:
        probe = getattr(float8, name, None)

        if probe is not None:
            probe.cache_clear()


@pytest.mark.parametrize("device", ("cuda", "mlu"))
def test_float8_probe_uses_device_specific_external_child(device, monkeypatch):
    control = CapabilityResult(supported=True)
    expected = CapabilityResult(False, "triton float8: unsupported lowering")
    calls = []

    def fake_run(name, source, *, timeout=60):
        calls.append((name, source, timeout))

        return control if "FP16 control" in name else expected

    monkeypatch.setattr(float8, "run_python_probe", fake_run)
    probe = getattr(float8, f"float8_e5m2_{device}")

    assert probe() is expected
    assert tuple(name for name, _, _ in calls) == (
        f"triton {device} FP16 control",
        f"triton {device} float8_e5m2",
    )
    assert f"DEVICE = {device!r}" in calls[0][1]
    assert "PHASE = 'control'" in calls[0][1]
    assert f"DEVICE = {device!r}" in calls[1][1]
    assert "PHASE = 'float8'" in calls[1][1]
    assert all("@triton.jit" in source for _, source, _ in calls)


def test_float8_probe_caches_per_device(monkeypatch):
    expected = CapabilityResult(supported=True)
    calls = []

    def fake_run(name, source, *, timeout=60):
        calls.append((name, source, timeout))

        return expected

    monkeypatch.setattr(float8, "run_python_probe", fake_run)

    for device in ("cuda", "mlu"):
        probe = getattr(float8, f"float8_e5m2_{device}")

        assert probe() is expected
        assert probe() is expected

    assert tuple(name for name, _, _ in calls) == (
        "triton cuda FP16 control",
        "triton cuda float8_e5m2",
        "triton mlu FP16 control",
        "triton mlu float8_e5m2",
    )


def test_float8_probe_fp16_control_failure_is_an_error():
    source = float8._source_for("cuda", "control").replace(
        "    run(torch.float16)",
        '    raise RuntimeError("control")',
        1,
    )

    with pytest.raises(RuntimeError, match="FP16 control failed: control"):
        run_python_probe("triton float8_e5m2", source)


@pytest.mark.parametrize("phase", ("control", "float8"))
def test_float8_probe_setup_failure_is_an_error(phase):
    source = float8._source_for("cuda", phase).replace(
        "directory = tempfile.TemporaryDirectory()",
        'raise RuntimeError("setup")',
        1,
    )

    with pytest.raises(RuntimeError, match="probe setup failed: setup"):
        run_python_probe("triton float8_e5m2", source)


def test_float8_probe_fp8_failure_is_unavailable():
    source = float8._source_for("cuda", "float8").replace(
        "        run(torch.float8_e5m2)",
        '        raise RuntimeError("float8")',
        1,
    )

    result = run_python_probe("triton float8_e5m2", source)

    assert result == CapabilityResult(
        supported=False,
        reason="triton float8_e5m2: float8",
    )


def _supported_source():
    return 'import json; print(json.dumps({"status": "supported"}))'


def test_float8_probe_control_abnormal_exit_is_an_error(monkeypatch):
    phases = []

    def source_for(device, phase):
        del device
        phases.append(phase)

        return "import os; os._exit(17)"

    monkeypatch.setattr(float8, "_source_for", source_for, raising=False)

    with pytest.raises(RuntimeError, match="code 17"):
        float8._run_float8_e5m2_probe("cuda", timeout=1)

    assert phases == ["control"]


def test_float8_probe_control_timeout_is_an_error(monkeypatch):
    def source_for(device, phase):
        del device, phase

        return "import time; time.sleep(10)"

    monkeypatch.setattr(float8, "_source_for", source_for, raising=False)

    with pytest.raises(RuntimeError, match="timed out"):
        float8._run_float8_e5m2_probe("cuda", timeout=0.01)


@pytest.mark.parametrize(
    "float8_source, reason",
    (
        ("import os; os._exit(19)", "code 19"),
        ("import time; time.sleep(10)", "timed out"),
    ),
)
def test_float8_probe_abnormal_exit_or_timeout_is_unavailable(
    float8_source,
    reason,
    monkeypatch,
):
    phases = []

    def source_for(device, phase):
        del device
        phases.append(phase)

        return _supported_source() if phase == "control" else float8_source

    monkeypatch.setattr(float8, "_source_for", source_for, raising=False)

    result = float8._run_float8_e5m2_probe("cuda", timeout=1)

    assert not result.supported
    assert reason in result.reason
    assert phases == ["control", "float8"]


def _parameter_capabilities(parameters):
    capabilities = {}

    for parameter in parameters:
        device, dtype, atol = parameter.values
        references = tuple(
            mark.args[0]
            for mark in parameter.marks
            if mark.name == "requires_capability"
        )
        capabilities[(device, dtype)] = (atol, references)

    return capabilities


def test_float8_parameters_use_device_specific_capabilities():
    import tests.test_matmul as matmul

    capabilities = _parameter_capabilities(
        matmul._device_dtype_config(("cuda", "mlu"), fp16_atol=0.25)
    )

    assert capabilities[("cuda", torch.float16)] == (0.25, ())
    assert capabilities[("mlu", torch.float16)] == (0.25, ())
    assert capabilities[("cuda", torch.float8_e5m2)] == (
        0.125,
        ("tests.capabilities.float8:float8_e5m2_cuda",),
    )
    assert capabilities[("mlu", torch.float8_e5m2)] == (
        0.125,
        ("tests.capabilities.float8:float8_e5m2_mlu",),
    )


def test_matmul_and_addmm_use_joint_device_dtype_parameters():
    import tests.test_addmm as addmm
    import tests.test_matmul as matmul

    for function in (matmul.test, addmm.test):
        parameter_names = tuple(
            mark.args[0] for mark in function.pytestmark if mark.name == "parametrize"
        )

        assert "device, dtype, atol" in parameter_names
        assert "device" not in parameter_names
        assert "dtype, atol" not in parameter_names
