"""Probe external materialization dependencies without importing NineToothed."""

import functools

from tests.capabilities.runner import run_python_probe

_CUDA_TOOLCHAIN_SOURCE = r"""
import ctypes
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def report(status, reason=""):
    print(json.dumps({"status": status, "reason": reason}))


try:
    candidates = (
        shutil.which("nvcc"),
        str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "nvcc"),
    )
    nvcc = next((candidate for candidate in candidates if candidate and Path(candidate).is_file()), None)

    if nvcc is None:
        report("unavailable", "CUDA toolchain probe failed: nvcc was not found.")
        raise SystemExit(0)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "capability_probe.cu"
        library = root / "capability_probe.so"
        source.write_text(
            'extern "C" __global__ void capability_probe() {}\n',
            encoding="utf-8",
        )
        completed = subprocess.run(
            [nvcc, "-shared", "-Xcompiler", "-fPIC", str(source), "-o", str(library)],
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            diagnostic = completed.stderr.strip() or completed.stdout.strip() or "no compiler diagnostic"
            report(
                "unavailable",
                f"CUDA toolchain probe failed: nvcc exited with code {completed.returncode}: {diagnostic}",
            )
            raise SystemExit(0)

        if not library.is_file():
            report(
                "unavailable",
                "CUDA toolchain probe failed: nvcc exited successfully but did not produce the shared library.",
            )
            raise SystemExit(0)

        ctypes.CDLL(str(library))

    report("supported")
except Exception as error:
    report(
        "unavailable",
        f"CUDA toolchain probe failed: {type(error).__name__}: {error}",
    )
"""


_TRITON_AOT_SOURCE = r"""
import importlib
import json


def report(status, reason=""):
    print(json.dumps({"status": status, "reason": reason}))


try:
    importlib.import_module("triton.backends.nvidia")
    importlib.import_module("triton.tools.compile")
    importlib.import_module("triton.tools.link")
except ImportError as error:
    report(
        "unavailable",
        f"Triton AOT probe failed: {type(error).__name__}: {error}",
    )
except Exception as error:
    report(
        "error",
        f"Triton AOT probe failed: {type(error).__name__}: {error}",
    )
else:
    report("supported")
"""


_TILELANG_CUDA_SOURCE = r"""
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path


def report(status, reason=""):
    print(json.dumps({"status": status, "reason": reason}))


try:
    import tilelang
    import tvm

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "tilelang_capability_probe.py"
        source.write_text(
            "import tilelang.language as T\n\n"
            "@T.prim_func\n"
            "def capability_probe(input: T.handle, output: T.handle):\n"
            "    input_buf = T.match_buffer(input, (1,), T.float32)\n"
            "    output_buf = T.match_buffer(output, (1,), T.float32)\n"
            "    with T.Kernel(1, threads=1) as block:\n"
            "        output_buf[block] = input_buf[block]\n",
            encoding="utf-8",
        )
        spec = importlib.util.spec_from_file_location(
            "tilelang_capability_probe",
            source,
        )

        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load the TileLang capability probe module.")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kernel = tilelang.compile(
            module.capability_probe,
            execution_backend="tvm_ffi",
            target="cuda",
        )
        bundled_library = getattr(kernel.adapter, "libpath", None)
        library_path = root / "capability_probe.so"

        if bundled_library and Path(bundled_library).is_file():
            shutil.copy2(bundled_library, library_path)
        else:
            kernel.export_library(str(library_path))

        if not library_path.is_file():
            report(
                "unavailable",
                "TileLang CUDA probe failed: compilation did not produce a shared library.",
            )
            raise SystemExit(0)

        tvm.runtime.load_module(str(library_path))

    report("supported")
except Exception as error:
    report(
        "unavailable",
        f"TileLang CUDA probe failed: {type(error).__name__}: {error}",
    )
"""


@functools.cache
def cuda_toolchain():
    return run_python_probe("CUDA toolchain", _CUDA_TOOLCHAIN_SOURCE)


@functools.cache
def triton_aot():
    cuda = cuda_toolchain()

    if not cuda.supported:
        return cuda

    return run_python_probe("Triton AOT", _TRITON_AOT_SOURCE)


@functools.cache
def tilelang_cuda():
    return run_python_probe("TileLang CUDA", _TILELANG_CUDA_SOURCE)


__all__ = ["cuda_toolchain", "tilelang_cuda", "triton_aot"]
