"""Shared compiler-toolchain discovery and command construction."""

import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

_CUDA_ARCH_PATTERN = re.compile(r"(?:sm|compute)_[0-9]{2,}[a-z]?")

_BANGC_ARCH_ALIASES = {
    "compute_30": "compute_30",
    "compute_50": "compute_50",
    "mlu370": "compute_30",
    "mtp_372": "compute_30",
    "mlu590": "compute_50",
    "mtp_592": "compute_50",
}


def normalize_bangc_arch(value: Any) -> str:
    """Return a validated cncc ``--bang-arch`` value."""
    if not isinstance(value, str):
        raise TypeError("The BangC `arch` backend option must be a string.")

    arch = value.strip().lower()

    if arch == "native":
        return arch

    resolved = _BANGC_ARCH_ALIASES.get(arch)

    if resolved is None:
        raise ValueError(
            "The BangC `arch` backend option must be `native`, `compute_50` "
            "(MLU590/mtp_592), or `compute_30` (MLU370/mtp_372)."
        )

    return resolved


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


def resolve_bangc_arch(arch: str) -> str:
    """Resolve `native` to a concrete cncc architecture value."""
    arch = normalize_bangc_arch(arch)

    if arch != "native":
        return arch

    explicit = os.environ.get("NINETOOTHED_BANGC_ARCH")

    if explicit:
        return normalize_bangc_arch(explicit)

    try:
        import torch

        if hasattr(torch, "mlu") and torch.mlu.is_available():
            capability = getattr(torch.mlu, "get_device_capability", None)

            if callable(capability):
                major = int(capability(0)[0])

                return "compute_30" if major < 5 else "compute_50"
    except (ImportError, RuntimeError, ValueError, IndexError, AttributeError):
        pass

    return "compute_50"


def bangc_compile_command(
    source: str | Path,
    output: str | Path,
    *,
    arch: str,
    cncc: str | None = None,
) -> tuple[str, ...]:
    """Construct the shared-library command used by the BangC materializer."""
    return (
        cncc or find_cncc(),
        "--shared",
        "-fPIC",
        "-O3",
        f"--bang-arch={normalize_bangc_arch(arch)}",
        str(source),
        "-o",
        str(output),
    )


def find_cncc() -> str:
    """Locate the Cambricon BangC compiler (cncc)."""
    explicit = os.environ.get("NINETOOTHED_BANGC_COMPILER")

    if explicit is not None:
        return _validated_compiler(explicit, explicit=True)

    neuware_roots = tuple(
        Path(value)
        for value in (
            os.environ.get("NEUWARE_HOME"),
            os.environ.get("NEUWARE_ROOT"),
        )
        if value
    )
    candidates = (
        shutil.which("cncc"),
        *(str(root / "bin" / "cncc") for root in neuware_roots),
        "/usr/local/neuware/bin/cncc",
    )

    for candidate in candidates:
        if candidate:
            try:
                return _validated_compiler(candidate, explicit=False)
            except RuntimeError:
                continue

    if bangc_remote_spec() is not None:
        raise RuntimeError(
            "The BangC compiler cncc is not installed locally; the remote "
            "toolchain will be used instead."
        )

    raise RuntimeError(
        "BangC backend requires the Cambricon cncc compiler; set "
        "NINETOOTHED_BANGC_COMPILER, NEUWARE_HOME, or configure a remote "
        "toolchain with NINETOOTHED_BANGC_SSH."
    )


def bangc_remote_spec() -> dict[str, Any] | None:
    """Return the remote BangC toolchain configuration, if any."""
    ssh = os.environ.get("NINETOOTHED_BANGC_SSH")

    if not ssh:
        return None

    user_host, _, port = ssh.rpartition(":")
    user_host = user_host or ssh
    port = port or "22"

    return {
        "user_host": user_host,
        "port": port,
        "container": os.environ.get("NINETOOTHED_BANGC_CONTAINER", ""),
        "remote_dir": os.environ.get(
            "NINETOOTHED_BANGC_REMOTE_DIR", "/tmp/ninetoothed-bangc"
        ),
    }


def bangc_remote_base_command(spec: dict[str, Any]) -> tuple[str, ...]:
    """Build the SSH prefix that lands a shell on the cncc host.

    ``docker exec -i <c> bash -s`` executes the script streamed over stdin.
    Passing the script through argv is unreliable on some Windows OpenSSH
    builds (arguments can be dropped before reaching the container), while
    the ``bash -s`` stdin channel is stable across platforms.
    """
    command = ["ssh", "-p", str(spec["port"]), str(spec["user_host"])]

    if spec.get("container"):
        command += ["docker", "exec", "-i", str(spec["container"]), "bash", "-s"]

    return tuple(command)


def bangc_compiler_identity(*, required: bool = False) -> dict[str, Any]:
    """Return the resolved BangC compiler identity used in cache keys."""
    spec = bangc_remote_spec()

    if spec is not None:
        return _bangc_remote_compiler_identity(
            str(spec["user_host"]), str(spec["port"]), str(spec.get("container", ""))
        )

    try:
        compiler = find_cncc()
    except RuntimeError:
        if required or os.environ.get("NINETOOTHED_BANGC_COMPILER") is not None:
            raise
        return {"available": False}

    path = Path(compiler).resolve(strict=True)
    stat = path.stat()

    return _cncc_identity(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=32)
def _bangc_remote_compiler_identity(
    user_host: str, port: str, container: str
) -> dict[str, Any]:
    spec = {
        "user_host": user_host,
        "port": port,
        "container": container,
        "remote_dir": os.environ.get(
            "NINETOOTHED_BANGC_REMOTE_DIR", "/tmp/ninetoothed-bangc"
        ),
    }
    command = (*bangc_remote_base_command(spec),)
    script = "cncc --version 2>&1 | grep -m1 cncc\n"

    try:
        result = subprocess.run(
            command,
            input=script.encode("utf-8"),
            check=False,
            capture_output=True,
            timeout=30,
        )
        version_lines = [
            line.strip()
            for line in result.stdout.decode("utf-8", errors="replace").splitlines()
            + result.stderr.decode("utf-8", errors="replace").splitlines()
            if line.strip() and "cncc" in line
        ]
        version_text = version_lines[0] if version_lines else ""
    except (OSError, subprocess.SubprocessError):
        version_text = ""

    return {
        "available": True,
        "remote": f"{user_host}:{port}",
        "container": container,
        "version": version_text or "unavailable",
    }


@lru_cache(maxsize=32)
def _cncc_identity(path: str, size: int, mtime_ns: int) -> dict[str, Any]:
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
            f"Failed to query BangC compiler identity from `{path}`."
        ) from exc

    version_text = "\n".join(
        value.strip() for value in (result.stdout, result.stderr) if value.strip()
    )

    if result.returncode != 0 or not version_text:
        raise RuntimeError(
            f"BangC compiler `{path}` did not provide a usable --version result."
        )

    return {
        "available": True,
        "path": path,
        "size": size,
        "mtime_ns": mtime_ns,
        "version": version_text,
    }


__all__ = [
    "bangc_compile_command",
    "bangc_compiler_identity",
    "bangc_remote_base_command",
    "bangc_remote_spec",
    "cuda_compiler_identity",
    "cuda_compile_command",
    "cuda_compute_capability",
    "find_cncc",
    "find_nvcc",
    "normalize_bangc_arch",
    "normalize_cuda_arch",
]
