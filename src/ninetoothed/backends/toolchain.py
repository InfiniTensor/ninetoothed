"""Shared compiler-toolchain discovery and command construction."""

import os
import re
import shutil
from pathlib import Path
from typing import Any

_CUDA_ARCH_PATTERN = re.compile(r"(?:sm|compute)_[0-9]{2,}[a-z]?")


def normalize_cuda_arch(value: Any) -> str:
    """Return a validated NVCC architecture value."""
    if not isinstance(value, str):
        raise TypeError("The CUDA `arch` backend option must be a string.")

    arch = value.strip().lower()

    if arch != "native" and _CUDA_ARCH_PATTERN.fullmatch(arch) is None:
        raise ValueError(
            "The CUDA `arch` backend option must be `native`, `sm_<version>`, "
            "or `compute_<version>`."
        )

    return arch


def cuda_compute_capability(arch: str) -> str | None:
    """Infer a pass constraint value from a concrete NVCC architecture."""
    arch = normalize_cuda_arch(arch)

    if arch == "native":
        return None

    digits = re.search(r"[0-9]+", arch)
    assert digits is not None
    version = digits.group()

    return f"{int(version[:-1])}.{version[-1]}"


def find_nvcc() -> str:
    """Locate NVCC using PATH first and CUDA_HOME second."""
    candidates = (
        shutil.which("nvcc"),
        str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "nvcc"),
    )

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate

    raise RuntimeError("CUDA backend requires nvcc; set CUDA_HOME or add nvcc to PATH.")


def cuda_compile_command(
    source: str | Path,
    output: str | Path,
    *,
    arch: str,
    nvcc: str | None = None,
) -> tuple[str, ...]:
    """Construct the shared-library command used by the CUDA materializer."""
    return (
        nvcc or find_nvcc(),
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-O3",
        f"-arch={normalize_cuda_arch(arch)}",
        str(source),
        "-o",
        str(output),
    )


__all__ = [
    "cuda_compile_command",
    "cuda_compute_capability",
    "find_nvcc",
    "normalize_cuda_arch",
]
