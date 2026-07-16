import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ninetoothed.backends.core import Artifact, BuiltArtifact, Target
from ninetoothed.backends.materializers import triton as triton_materializer
from ninetoothed.compiler.runtime import KernelLaunchError
from ninetoothed.ir import LaunchABI


def _generated_source(name):
    return f"""
#include <cuda.h>

CUmodule {name}_mod = NULL;
CUfunction {name}_func = NULL;

void load_{name}() {{
    {name}_mod = NULL;
    {name}_func = NULL;
}}
"""


def test_triton_aot_kernel_names_require_matching_generated_symbols(tmp_path):
    first = tmp_path / "compiled.first.c"
    second = tmp_path / "compiled.second.c"
    first.write_text(_generated_source("guarded_first_ab12"), encoding="utf-8")
    second.write_text(_generated_source("guarded_second_cd34"), encoding="utf-8")

    assert triton_materializer._triton_aot_kernel_names((first, second)) == (
        "guarded_first_ab12",
        "guarded_second_cd34",
    )


@pytest.mark.parametrize(
    "source",
    (
        "CUmodule changed_mod = NULL;\nvoid load_changed() {}\n",
        (
            "CUmodule changed_mod = NULL;\n"
            "CUfunction another_func = NULL;\n"
            "void load_changed() {}\n"
        ),
        (
            "CUmodule changed_mod = NULL;\n"
            "CUfunction changed_func = NULL;\n"
            "static void load_changed() {}\n"
        ),
    ),
)
def test_triton_aot_kernel_names_reject_format_drift(tmp_path, source):
    generated = tmp_path / "compiled.changed.c"
    generated.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="Triton AOT source format"):
        triton_materializer._triton_aot_kernel_names((generated,))


def test_triton_context_guard_uses_context_state_instead_of_device_ordinals():
    source = triton_materializer._triton_context_guard_source(
        ("guarded_first_ab12", "guarded_second_cd34")
    )

    assert 'extern "C" CUmodule guarded_first_ab12_mod' in source
    assert 'extern "C" CUfunction guarded_second_cd34_func' in source
    assert 'extern "C" void load_guarded_first_ab12' in source
    assert 'extern "C" CUresult ninetoothed_triton_enter' in source
    assert 'extern "C" void ninetoothed_triton_leave' in source
    assert "cuCtxGetCurrent" in source
    assert "std::mutex" in source
    assert "std::unordered_map<CUcontext, State>" in source
    assert "cuCtxGetDevice" not in source
    assert "CUdevice" not in source


class _FakeExport:
    def __init__(self, name, events, *, result=0, error=None):
        self._name = name
        self._events = events
        self._result = result
        self._error = error
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        del args
        self._events.append(self._name)

        if self._error is not None:
            raise self._error
        return self._result


def _built_artifact(tmp_path):
    source = Artifact(
        backend=Target.TRITON,
        kernel_name="guarded",
        language="python/triton",
        sources={"guarded.py": ""},
    )

    return BuiltArtifact(
        source=source,
        cache_key="guarded-cache-key",
        source_path=str(tmp_path / "guarded.py"),
        binary_path=str(tmp_path / "guarded.triton.so"),
        manifest_path=str(tmp_path / "guarded.triton.manifest.json"),
        abi={},
    )


def _load_fake_artifact(
    monkeypatch, tmp_path, *, enter_result=0, kernel_result=0, error=None
):
    events = []
    library = SimpleNamespace(
        guarded_kernel_default=_FakeExport(
            "kernel", events, result=kernel_result, error=error
        ),
        ninetoothed_triton_enter=_FakeExport("enter", events, result=enter_result),
        ninetoothed_triton_leave=_FakeExport("leave", events),
    )
    monkeypatch.setattr(triton_materializer.ctypes, "CDLL", lambda path: library)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(cuda_stream=17),
    )
    launch = triton_materializer.TritonMaterializer().load_built_artifact(
        _built_artifact(tmp_path)
    )

    return launch, library, events


def test_fresh_aot_materialization_enters_launches_and_leaves(monkeypatch, tmp_path):
    events = []
    library = SimpleNamespace(
        fresh_guarded_kernel_default=_FakeExport("kernel", events),
        ninetoothed_triton_enter=_FakeExport("enter", events),
        ninetoothed_triton_leave=_FakeExport("leave", events),
    )
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    cache_library = cache_directory / "fresh_guarded.triton.so"
    cache_library.write_bytes(b"library")
    cache_library.with_suffix(".manifest.json").write_text(
        json.dumps({"triton_aot_launcher_schema": 1}),
        encoding="utf-8",
    )
    source = tmp_path / "fresh_guarded.py"
    published = tmp_path / "fresh_guarded.published.so"

    class FakeHandle:
        def __init__(self, compilation, function, launch, source, library_path):
            del compilation, function, source
            self._launch = launch
            self._built_artifact = SimpleNamespace(
                manifest_path=str(Path(library_path).with_suffix(".manifest.json"))
            )

        def __call__(self, *args, **kwargs):
            return self._launch(*args, **kwargs)

    from ninetoothed.compiler import runtime

    monkeypatch.setattr(triton_materializer.ctypes, "CDLL", lambda path: library)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(cuda_stream=17),
    )
    monkeypatch.setattr(
        triton_materializer, "compilation_cache_key", lambda compilation: "fresh-key"
    )
    monkeypatch.setattr(
        triton_materializer,
        "write_source",
        lambda *args, **kwargs: source,
    )
    monkeypatch.setattr(
        triton_materializer,
        "artifact_directory",
        lambda cache_key: cache_directory,
    )
    monkeypatch.setattr(
        triton_materializer,
        "cache_lock",
        lambda path: nullcontext(),
    )
    monkeypatch.setattr(triton_materializer, "write_manifest", lambda *args: None)
    monkeypatch.setattr(runtime, "Handle", FakeHandle)
    monkeypatch.setattr(runtime, "_built_manifest", lambda *args: {})
    monkeypatch.setattr(
        runtime,
        "_publish_library",
        lambda *args: published,
    )
    compilation = SimpleNamespace(
        artifact=SimpleNamespace(
            kernel_name="fresh_guarded",
            primary_source="",
        ),
        launch_abi=LaunchABI(),
        kernel=SimpleNamespace(tensors=()),
    )

    handle = triton_materializer._aot_materialize(compilation, output_dir=tmp_path)

    assert handle() is None
    assert events == ["enter", "kernel", "leave"]
    assert library.ninetoothed_triton_enter.argtypes == []
    assert library.ninetoothed_triton_enter.restype is triton_materializer.ctypes.c_int
    assert library.ninetoothed_triton_leave.argtypes == []
    assert library.ninetoothed_triton_leave.restype is None


def test_reloaded_aot_wrapper_enters_launches_and_leaves(monkeypatch, tmp_path):
    launch, library, events = _load_fake_artifact(monkeypatch, tmp_path)

    assert launch() is None
    assert events == ["enter", "kernel", "leave"]
    assert library.ninetoothed_triton_enter.argtypes == []
    assert library.ninetoothed_triton_enter.restype is triton_materializer.ctypes.c_int
    assert library.ninetoothed_triton_leave.argtypes == []
    assert library.ninetoothed_triton_leave.restype is None


@pytest.mark.parametrize(
    "kernel_result,error,exception",
    (
        (17, None, KernelLaunchError),
        (0, RuntimeError("launcher failed"), RuntimeError),
    ),
)
def test_reloaded_aot_wrapper_leaves_after_kernel_failures(
    monkeypatch,
    tmp_path,
    kernel_result,
    error,
    exception,
):
    launch, _, events = _load_fake_artifact(
        monkeypatch,
        tmp_path,
        kernel_result=kernel_result,
        error=error,
    )

    with pytest.raises(exception):
        launch()

    assert events == ["enter", "kernel", "leave"]


def test_reloaded_aot_wrapper_does_not_launch_when_enter_fails(monkeypatch, tmp_path):
    launch, _, events = _load_fake_artifact(
        monkeypatch,
        tmp_path,
        enter_result=201,
    )

    with pytest.raises(KernelLaunchError):
        launch()

    assert events == ["enter"]


def test_reloaded_unguarded_artifact_requires_rebuild(monkeypatch, tmp_path):
    library = SimpleNamespace(
        guarded_kernel_default=_FakeExport("kernel", []),
    )
    monkeypatch.setattr(triton_materializer.ctypes, "CDLL", lambda path: library)

    with pytest.raises(RuntimeError, match="rebuild"):
        triton_materializer.TritonMaterializer().load_built_artifact(
            _built_artifact(tmp_path)
        )


@pytest.mark.parametrize("manifest_state", ("missing", "stale", "current"))
def test_stale_triton_launcher_schema_triggers_rebuild(
    monkeypatch,
    tmp_path,
    manifest_state,
):
    library = tmp_path / "guarded.triton.so"
    library.write_bytes(b"old library")
    manifest = library.with_suffix(".manifest.json")

    if manifest_state != "missing":
        schema = (
            triton_materializer._TRITON_AOT_LAUNCHER_SCHEMA
            if manifest_state == "current"
            else triton_materializer._TRITON_AOT_LAUNCHER_SCHEMA - 1
        )
        manifest.write_text(
            json.dumps({"triton_aot_launcher_schema": schema}),
            encoding="utf-8",
        )

    compile_calls = []
    monkeypatch.setattr(
        triton_materializer,
        "_compile_aot_library",
        lambda compilation, source, output: compile_calls.append(
            (compilation, source, output)
        ),
    )
    compilation = object()
    source = tmp_path / "guarded.py"

    triton_materializer._ensure_aot_library(compilation, source, library)

    assert bool(compile_calls) is (manifest_state != "current")


def test_aot_compilation_links_generated_context_guard(monkeypatch, tmp_path):
    commands = []
    guard_sources = []

    def run(command, *, check, env=None):
        del check, env
        command = tuple(str(argument) for argument in command)
        commands.append(command)

        if "triton.tools.compile" in command:
            output_base = Path(command[command.index("--out-path") + 1])
            generated = output_base.parent / "compiled.ab12.c"
            generated.write_text(_generated_source("guarded_ab12"), encoding="utf-8")
            output_base.parent.joinpath("compiled.ab12.h").write_text(
                "void guarded_ab12(void);\n",
                encoding="utf-8",
            )
        elif "triton.tools.link" in command:
            output_base = Path(command[command.index("--out") + 1])
            output_base.with_suffix(".c").write_text("", encoding="utf-8")
        elif command[0] == "nvcc":
            guard_paths = tuple(
                Path(argument)
                for argument in command
                if argument.endswith("ninetoothed_triton_context_guard.cu")
            )
            guard_sources.extend(
                path.read_text(encoding="utf-8") for path in guard_paths
            )
            Path(command[command.index("-o") + 1]).write_bytes(b"library")

        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(triton_materializer.subprocess, "run", run)
    monkeypatch.setattr(triton_materializer, "find_nvcc", lambda: "nvcc")
    monkeypatch.setattr(triton_materializer, "_ensure_c_compiler", lambda: None)
    monkeypatch.setattr(
        triton_materializer, "_compile_signature", lambda compilation: "*fp32,1"
    )
    monkeypatch.setattr(
        triton_materializer, "_compile_grid", lambda compilation: "1,1,1"
    )
    monkeypatch.setattr(
        triton_materializer, "_compile_schedule", lambda compilation: (4, 3)
    )
    compilation = SimpleNamespace(
        artifact=SimpleNamespace(kernel_name="guarded"),
    )
    source = tmp_path / "guarded.py"
    source.write_text("", encoding="utf-8")
    library = tmp_path / "guarded.triton.so"

    triton_materializer._compile_aot_library(compilation, source, library)

    nvcc_command = next(command for command in commands if command[0] == "nvcc")
    assert any(
        argument.endswith("ninetoothed_triton_context_guard.cu")
        for argument in nvcc_command
    )
    assert any(
        nvcc_command[index : index + 4] == ("-Xlinker", "-z", "-Xlinker", "defs")
        for index in range(len(nvcc_command) - 3)
    )
    assert len(guard_sources) == 1
    assert "ninetoothed_triton_enter" in guard_sources[0]
    assert "ninetoothed_triton_leave" in guard_sources[0]
