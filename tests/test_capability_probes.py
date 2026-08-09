import dataclasses
import json
import textwrap
from types import SimpleNamespace

import pytest

import tests.capabilities as capabilities
import tests.conftest as suite_conftest
from tests.capabilities import materialization
from tests.capabilities.model import CapabilityResult
from tests.capabilities.runner import resolve_probe, run_python_probe


def resolved_probe():
    return CapabilityResult(supported=True)


def _protocol_source(payload):
    return f"import json; print(json.dumps({payload!r}))"


def test_capabilities_package_exports_public_protocol():
    assert capabilities.__all__ == (
        "CapabilityResult",
        "resolve_probe",
        "run_python_probe",
    )
    assert capabilities.CapabilityResult is CapabilityResult
    assert capabilities.resolve_probe is resolve_probe
    assert capabilities.run_python_probe is run_python_probe


@pytest.fixture(autouse=True)
def clear_materialization_probe_caches():
    probes = (
        materialization.cuda_toolchain,
        materialization.triton_aot,
        materialization.tilelang_cuda,
    )

    for probe in probes:
        probe.cache_clear()

    yield

    for probe in probes:
        probe.cache_clear()


def test_capability_result_is_frozen():
    result = CapabilityResult(supported=True, reason="")

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.supported = False


def test_capability_result_reason_defaults_to_empty():
    assert CapabilityResult(supported=True) == CapabilityResult(
        supported=True,
        reason="",
    )


def test_supported_probe_uses_final_nonempty_json_line():
    source = "\n".join(
        (
            "import json",
            'print("compiler diagnostic")',
            'print("")',
            'print(json.dumps({"status": "supported"}))',
        )
    )

    result = run_python_probe("sample", source)

    assert result == CapabilityResult(supported=True, reason="")


def test_unavailable_probe_preserves_reason():
    result = run_python_probe(
        "sample",
        _protocol_source(
            {"status": "unavailable", "reason": "The compiler is unavailable."}
        ),
    )

    assert result == CapabilityResult(
        supported=False,
        reason="sample: The compiler is unavailable.",
    )


def test_error_probe_raises_runtime_error():
    source = _protocol_source(
        {"status": "error", "reason": "The probe implementation failed."}
    )

    with pytest.raises(RuntimeError, match="probe implementation failed"):
        run_python_probe("sample", source)


def test_probe_timeout_raises_runtime_error():
    with pytest.raises(RuntimeError, match="timed out"):
        run_python_probe("sample", "import time; time.sleep(10)", timeout=0.01)


@pytest.mark.parametrize(
    "source",
    (
        'print("not json")',
        f"print({json.dumps(json.dumps(['supported']))!r})",
        _protocol_source({"status": "unknown"}),
        _protocol_source({"status": "unavailable"}),
    ),
)
def test_invalid_probe_output_raises_runtime_error(source):
    with pytest.raises(RuntimeError, match="invalid"):
        run_python_probe("sample", source)


def test_abnormal_probe_exit_is_unavailable():
    source = "\n".join(
        (
            "import sys",
            'sys.stderr.write("native compiler aborted\\n")',
            "raise SystemExit(17)",
        )
    )

    result = run_python_probe("sample", source)

    assert not result.supported
    assert "code 17" in result.reason
    assert "native compiler aborted" in result.reason


def test_resolve_probe_executes_fully_qualified_reference():
    result = resolve_probe("tests.test_capability_probes:resolved_probe")

    assert result == CapabilityResult(supported=True)


def test_requires_capability_marker_is_registered():
    registered = []
    config = SimpleNamespace(
        addinivalue_line=lambda name, value: registered.append((name, value))
    )

    suite_conftest.pytest_configure(config)

    assert registered == [
        (
            "markers",
            "requires_capability(reference): require an external test capability",
        )
    ]


def test_requires_capability_stops_at_first_unavailable_marker(monkeypatch):
    references = ("example:first", "example:second")
    markers = tuple(
        pytest.mark.requires_capability(reference).mark for reference in references
    )
    item = SimpleNamespace(iter_markers=lambda name: markers)
    calls = []

    def resolve(reference):
        calls.append(reference)

        return CapabilityResult(
            supported=reference == references[1],
            reason="The first capability is unavailable.",
        )

    monkeypatch.setattr(suite_conftest, "resolve_probe", resolve)

    with pytest.raises(pytest.skip.Exception, match="first capability"):
        suite_conftest.pytest_runtest_setup(item)

    assert calls == [references[0]]


def test_supported_capability_does_not_catch_later_failure(monkeypatch):
    marker = pytest.mark.requires_capability("example:supported").mark
    item = SimpleNamespace(iter_markers=lambda name: (marker,))
    monkeypatch.setattr(
        suite_conftest,
        "resolve_probe",
        lambda reference: CapabilityResult(supported=True),
    )

    suite_conftest.pytest_runtest_setup(item)

    with pytest.raises(RuntimeError, match="NineToothed failed"):
        raise RuntimeError("NineToothed failed after the external probe passed.")


def test_probe_errors_are_not_converted_to_skips(monkeypatch):
    marker = pytest.mark.requires_capability("example:error").mark
    item = SimpleNamespace(iter_markers=lambda name: (marker,))

    def fail(reference):
        del reference
        raise RuntimeError("The probe protocol failed.")

    monkeypatch.setattr(suite_conftest, "resolve_probe", fail)

    with pytest.raises(RuntimeError, match="probe protocol failed"):
        suite_conftest.pytest_runtest_setup(item)


@pytest.mark.parametrize(
    "probe_name, runner_name",
    (
        ("cuda_toolchain", "CUDA toolchain"),
        ("triton_aot", "Triton AOT"),
        ("tilelang_cuda", "TileLang CUDA"),
    ),
)
def test_materialization_probe_repeated_calls_use_cache(
    probe_name, runner_name, monkeypatch
):
    probe = getattr(materialization, probe_name)
    probe.cache_clear()
    calls = []
    supported = CapabilityResult(supported=True)

    def run(name, source, timeout=60):
        calls.append((name, source, timeout))

        return supported

    monkeypatch.setattr(materialization, "run_python_probe", run)

    if probe_name == "triton_aot":
        monkeypatch.setattr(
            materialization,
            "cuda_toolchain",
            lambda: CapabilityResult(supported=True),
        )

    assert probe() is supported
    assert probe() is supported
    assert tuple(name for name, _, _ in calls) == (runner_name,)


def test_cuda_toolchain_rejects_compiler_that_creates_no_artifact(
    monkeypatch, tmp_path
):
    compiler = tmp_path / "nvcc"
    compiler.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    compiler.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    materialization.cuda_toolchain.cache_clear()

    result = materialization.cuda_toolchain()

    assert not result.supported
    assert "did not produce" in result.reason


def test_triton_aot_short_circuits_when_cuda_toolchain_is_unavailable(monkeypatch):
    materialization.triton_aot.cache_clear()
    unavailable = CapabilityResult(
        supported=False,
        reason="The CUDA toolchain is unavailable.",
    )
    monkeypatch.setattr(materialization, "cuda_toolchain", lambda: unavailable)

    def unexpected_probe(*args, **kwargs):
        pytest.fail("Triton tools were probed without a CUDA toolchain.")

    monkeypatch.setattr(materialization, "run_python_probe", unexpected_probe)

    assert materialization.triton_aot() is unavailable


def test_triton_aot_propagates_unexpected_import_failure(
    monkeypatch,
    tmp_path,
):
    triton = tmp_path / "triton"
    triton.mkdir()
    triton.joinpath("__init__.py").write_text("", encoding="utf-8")
    backends = triton / "backends"
    backends.mkdir()
    backends.joinpath("__init__.py").write_text("", encoding="utf-8")
    backends.joinpath("nvidia.py").write_text(
        'raise RuntimeError("unexpected Triton import failure")\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(
        materialization,
        "cuda_toolchain",
        lambda: CapabilityResult(supported=True),
    )
    materialization.triton_aot.cache_clear()

    with pytest.raises(
        RuntimeError,
        match="Capability probe 'Triton AOT' failed.*unexpected Triton import failure",
    ):
        materialization.triton_aot()


def test_triton_aot_reports_missing_backend_as_unavailable(monkeypatch, tmp_path):
    triton = tmp_path / "triton"
    triton.mkdir()
    triton.joinpath("__init__.py").write_text("", encoding="utf-8")
    backends = triton / "backends"
    backends.mkdir()
    backends.joinpath("__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(
        materialization,
        "cuda_toolchain",
        lambda: CapabilityResult(supported=True),
    )
    materialization.triton_aot.cache_clear()

    result = materialization.triton_aot()

    assert not result.supported
    assert "Triton AOT:" in result.reason
    assert "ModuleNotFoundError" in result.reason


def test_tilelang_native_abort_is_reported_as_unavailable(monkeypatch, tmp_path):
    package = tmp_path / "tilelang"
    package.mkdir()
    package.joinpath("__init__.py").write_text(
        "import os\nos.abort()\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    materialization.tilelang_cuda.cache_clear()

    result = materialization.tilelang_cuda()

    assert not result.supported
    assert "exited with code" in result.reason


@pytest.mark.parametrize(
    "library_mode, expected_events",
    (
        ("adapter", ("compile", "load")),
        ("export", ("compile", "export", "load")),
    ),
)
def test_tilelang_probe_compiles_exports_and_loads(
    library_mode, expected_events, monkeypatch, tmp_path
):
    events = tmp_path / "events.txt"
    tilelang = tmp_path / "tilelang"
    tilelang.mkdir()
    tilelang.joinpath("language.py").write_text(
        "import inspect\n"
        "handle = object()\n"
        "def prim_func(function):\n"
        "    inspect.getsource(function)\n"
        "    return function\n",
        encoding="utf-8",
    )
    tilelang.joinpath("__init__.py").write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            from . import language


            def _event(name):
                with open(os.environ["CAPABILITY_EVENTS"], "a", encoding="utf-8") as output:
                    output.write(f"{name}\\n")


            class Adapter:
                libpath = __file__ if os.environ["FAKE_TILELANG_MODE"] == "adapter" else None


            class Kernel:
                adapter = Adapter()

                def export_library(self, path):
                    _event("export")
                    Path(path).write_bytes(b"library")


            def compile(function, *, execution_backend, target):
                del function, execution_backend, target
                _event("compile")
                return Kernel()
            """
        ).lstrip(),
        encoding="utf-8",
    )
    tvm = tmp_path / "tvm"
    tvm.mkdir()
    tvm.joinpath("__init__.py").write_text(
        "from . import runtime\n",
        encoding="utf-8",
    )
    tvm.joinpath("runtime.py").write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path


            def load_module(path):
                adapter_library = os.environ["FAKE_TILELANG_ADAPTER_LIBRARY"]

                if (
                    os.environ["FAKE_TILELANG_MODE"] == "adapter"
                    and Path(path).resolve() == Path(adapter_library).resolve()
                ):
                    raise RuntimeError("loaded the adapter-owned library directly")

                if os.environ["FAKE_TILELANG_MODE"] == "adapter":
                    loaded = Path(path)
                    adapter = Path(adapter_library)

                    if loaded.read_bytes() != adapter.read_bytes():
                        raise RuntimeError("adapter library contents were not copied")

                    if loaded.stat().st_mtime_ns != adapter.stat().st_mtime_ns:
                        raise RuntimeError("adapter library metadata was not copied")

                with open(os.environ["CAPABILITY_EVENTS"], "a", encoding="utf-8") as output:
                    output.write("load\\n")
                return object()
            """
        ).lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("CAPABILITY_EVENTS", str(events))
    monkeypatch.setenv("FAKE_TILELANG_MODE", library_mode)
    monkeypatch.setenv("FAKE_TILELANG_ADAPTER_LIBRARY", str(tilelang / "__init__.py"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    materialization.tilelang_cuda.cache_clear()

    result = materialization.tilelang_cuda()

    assert result.supported
    assert tuple(events.read_text(encoding="utf-8").splitlines()) == expected_events


def test_materialization_probe_import_does_not_import_ninetoothed():
    source = "\n".join(
        (
            "import json",
            "import sys",
            "import tests.capabilities.materialization",
            "loaded = any(name == 'ninetoothed' or name.startswith('ninetoothed.') for name in sys.modules)",
            "status = 'error' if loaded else 'supported'",
            "reason = 'The probe imported NineToothed.' if loaded else ''",
            "print(json.dumps({'status': status, 'reason': reason}))",
        )
    )

    assert run_python_probe("materialization import", source).supported


def _capability_references(marks):
    return tuple(mark.args[0] for mark in marks if mark.name == "requires_capability")


def _parameter_capabilities(function, parameter):
    parametrize = next(
        mark
        for mark in function.pytestmark
        if mark.name == "parametrize" and mark.args[0] == parameter
    )

    return {
        value.values[0]: _capability_references(value.marks)
        for value in parametrize.args[1]
    }


def test_aot_gpu_tests_require_triton_aot_capability():
    import tests.test_aot as aot
    import tests.test_aot_auto_tuning as aot_auto_tuning

    reference = "tests.capabilities.materialization:triton_aot"
    gpu_tests = (
        "test_add",
        "test_addmm",
        "test_attention",
        "test_matmul",
        "test_conv2d",
        "test_fp32_scalar",
        "test_aot_with_static_non_power_of_two_innermost_sizes",
    )

    for name in gpu_tests:
        assert reference in _capability_references(getattr(aot, name).pytestmark)

    assert not _capability_references(
        getattr(aot.test_overflow_terms, "pytestmark", ())
    )
    assert reference in _capability_references(
        getattr(aot_auto_tuning.test_auto_tuning, "pytestmark", ())
    )


def test_reload_backend_parameters_require_matching_capabilities():
    import tests.test_built_artifact_reload as reload_tests

    prefix = "tests.capabilities.materialization:"

    assert _parameter_capabilities(
        reload_tests.test_aot_built_artifact_can_be_reloaded,
        "backend",
    ) == {
        "triton": (f"{prefix}triton_aot",),
        "cuda": (f"{prefix}cuda_toolchain",),
        "tilelang": (f"{prefix}tilelang_cuda",),
    }


def test_multi_context_and_cuda_empty_modes_require_capabilities():
    import tests.test_built_artifact_reload as reload_tests

    prefix = "tests.capabilities.materialization:"
    multi_context = _capability_references(
        reload_tests.test_triton_aot_handle_is_reusable_across_cuda_contexts.pytestmark
    )

    assert multi_context == (f"{prefix}triton_aot",)
    assert _parameter_capabilities(
        reload_tests.test_cuda_empty_tensor_is_a_no_op,
        "mode",
    ) == {
        "jit": (f"{prefix}cuda_toolchain",),
        "aot": (f"{prefix}cuda_toolchain",),
    }
