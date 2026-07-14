"""Content-addressed, process-safe compiler artifact cache."""

import hashlib
import json
import os
import platform
import tempfile
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator, Mapping

from ninetoothed.ir import ir_to_dict

_CACHE_ROOT = Path(
    os.environ.get("NINETOOTHED_CACHE_DIR", Path.home() / ".ninetoothed")
)
_WORKER = os.environ.get("PYTEST_XDIST_WORKER")
CACHE_DIR = _CACHE_ROOT / "xdist" / _WORKER if _WORKER else _CACHE_ROOT
TRITON_CACHE_DIR = _CACHE_ROOT / "triton"
TOOLCHAIN_LOCK_DIR = _CACHE_ROOT / "toolchains"


def stable_digest(value: Any) -> str:
    """Hash compiler state using a deterministic JSON representation."""
    payload = json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def compilation_cache_key(compilation) -> str:
    """Return a key covering source, IR, ABI, target, options, and toolchain."""
    request = compilation.request
    artifact = compilation.artifact

    return stable_digest(
        {
            "schema": 2,
            "source": artifact.sources,
            "ssa": ir_to_dict(compilation.kernel.ssa),
            "launch_plan": ir_to_dict(compilation.launch_plan),
            "backend": artifact.backend.value,
            "compiler_options": compilation.kernel.compiler_options,
            "backend_options": request.backend_options,
            "pipeline": request.pipeline,
            "pass_options": request.pass_options,
            "architecture": _architecture(compilation),
            "versions": _compiler_versions(),
        }
    )


def artifact_directory(key: str) -> Path:
    return _CACHE_ROOT / "artifacts" / key


def write_source(
    name: str,
    source: str,
    suffix: str,
    *,
    cache_key: str | None = None,
) -> Path:
    key = cache_key or stable_digest({"source": source, "suffix": suffix})
    path = artifact_directory(key) / f"{name}.{suffix}"
    atomic_write_text(path, source)

    return path


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    path = Path(path)
    atomic_write_text(
        path,
        json.dumps(_json_value(manifest), indent=2, sort_keys=True) + "\n",
    )

    return path


def read_manifest(path: str | Path) -> Mapping[str, Any] | None:
    path = Path(path)

    if not path.is_file():
        return None

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def atomic_write_text(path: str | Path, content: str) -> None:
    path = Path(path)

    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return

    _atomic_write(path, content.encode("utf-8"))


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    path = Path(path)

    if path.is_file() and path.read_bytes() == content:
        return

    _atomic_write(path, content)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with cache_lock(path):
        if path.is_file() and path.read_bytes() == content:
            return

        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )

        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())

            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


@contextmanager
def cache_lock(path: str | Path) -> Iterator[None]:
    """Hold a process lock associated with one cache artifact path."""
    lock_path = Path(f"{Path(path)}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+b") as lock:
        _lock_file(lock)

        try:
            yield
        finally:
            _unlock_file(lock)


def _lock_file(file) -> None:
    if os.name == "nt":
        import msvcrt

        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_EX)


def _unlock_file(file) -> None:
    if os.name == "nt":
        import msvcrt

        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def _compiler_versions() -> Mapping[str, str]:
    result = {"python": platform.python_version()}

    for package in ("ninetoothed", "triton", "tilelang"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def _architecture(compilation) -> Mapping[str, Any]:
    backend = compilation.artifact.backend.value
    backend_options = dict(compilation.request.backend_options or {})
    architecture = {
        "machine": platform.machine(),
        "cuda_arch": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    if backend == "cuda":
        target = str(backend_options.get("arch", "native"))
        architecture["cuda_target"] = target

        if target != "native":
            return architecture

    if backend in {"cuda", "tilelang", "triton"}:
        architecture.update(_runtime_cuda_architecture())
    return architecture


def _runtime_cuda_architecture() -> Mapping[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {
            "cuda_current_capability": None,
            "cuda_visible_capabilities": (),
        }

    def capability(device: int) -> str:
        major, minor = torch.cuda.get_device_capability(device)

        return f"sm_{major}{minor}"

    current = capability(torch.cuda.current_device())
    visible = tuple(
        dict.fromkeys(capability(device) for device in range(torch.cuda.device_count()))
    )

    return {
        "cuda_current_capability": current,
        "cuda_visible_capabilities": visible,
    }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}

    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]

    converted = ir_to_dict(value)

    if converted is not value:
        return _json_value(converted)
    return repr(value)


__all__ = [
    "CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TOOLCHAIN_LOCK_DIR",
    "artifact_directory",
    "atomic_write_bytes",
    "atomic_write_text",
    "cache_lock",
    "compilation_cache_key",
    "read_manifest",
    "stable_digest",
    "write_manifest",
    "write_source",
]
