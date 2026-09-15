"""Shared compiler-toolchain discovery and command construction."""

import os
import re
import shutil
import subprocess
from functools import lru_cache
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
    """Locate a CUDA-compatible compiler without hiding vendor executables."""
    explicit = os.environ.get("NINETOOTHED_CUDA_COMPILER")

    if explicit is not None:
        return _validated_compiler(explicit, explicit=True)

    cuda_roots = tuple(
        Path(value)
        for value in (
            os.environ.get("CUDA_HOME"),
            os.environ.get("CUDA_PATH"),
            os.environ.get("CUCC_PATH"),
        )
        if value
    )
    candidates = (
        shutil.which("nvcc"),
        shutil.which("cucc"),
        *(
            str(root / "bin" / compiler)
            for root in cuda_roots
            for compiler in ("nvcc", "cucc")
        ),
        "/usr/local/cuda/bin/nvcc",
    )

    for candidate in candidates:
        if candidate:
            try:
                return _validated_compiler(candidate, explicit=False)
            except RuntimeError:
                continue

    raise RuntimeError(
        "CUDA backend requires nvcc or a compatible vendor compiler; set "
        "NINETOOTHED_CUDA_COMPILER, CUDA_HOME, CUDA_PATH, or CUCC_PATH, or add "
        "nvcc/cucc to PATH."
    )


def cuda_compiler_identity(*, required: bool = False) -> dict[str, Any]:
    """Return the resolved CUDA compiler path and stable version identity."""
    try:
        compiler = find_nvcc()
    except RuntimeError:
        if required or os.environ.get("NINETOOTHED_CUDA_COMPILER") is not None:
            raise
        return {"available": False}

    path = Path(compiler).resolve(strict=True)
    stat = path.stat()

    return _cuda_compiler_identity(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=32)
def _cuda_compiler_identity(path: str, size: int, mtime_ns: int) -> dict[str, Any]:
    try:
        result = subprocess.run(
            (path, "--version"),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"Failed to query CUDA compiler identity from `{path}`."
        ) from exc

    version_text = "\n".join(
        value.strip() for value in (result.stdout, result.stderr) if value.strip()
    )

    if result.returncode != 0 or not version_text:
        raise RuntimeError(
            f"CUDA compiler `{path}` did not provide a usable --version result."
        )

    return {
        "available": True,
        "path": path,
        "size": size,
        "mtime_ns": mtime_ns,
        "version": version_text,
    }


def _validated_compiler(value: str, *, explicit: bool) -> str:
    path = Path(value).expanduser()

    if not path.is_file():
        if explicit:
            raise RuntimeError(f"Explicit CUDA compiler `{path}` is not a file.")

        raise RuntimeError(f"CUDA compiler `{path}` is not a file.")

    if os.name != "nt" and not os.access(path, os.X_OK):
        if explicit:
            raise RuntimeError(f"Explicit CUDA compiler `{path}` is not executable.")

        raise RuntimeError(f"CUDA compiler `{path}` is not executable.")

    return str(path)


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
    "cuda_compiler_identity",
    "cuda_compile_command",
    "cuda_compute_capability",
    "find_nvcc",
    "normalize_cuda_arch",
]
