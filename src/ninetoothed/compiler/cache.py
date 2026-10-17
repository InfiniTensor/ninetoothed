"""Content-addressed, process-safe compiler artifact cache."""

import hashlib
import json
import os
import platform
import tempfile
from contextlib import contextmanager
from dataclasses import fields
from dis import get_instructions
from functools import partial
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from inspect import getclosurevars, getsourcefile, isfunction
from pathlib import Path
from types import CodeType, ModuleType
from typing import Any, Iterator, Mapping

from ninetoothed.backends.toolchain import cuda_compiler_identity
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
            "schema": 3,
            "source": artifact.sources,
            "ssa": ir_to_dict(compilation.kernel.ssa),
            "launch_plan": ir_to_dict(compilation.launch_plan),
            "backend": artifact.backend.value,
            "target": _target_metadata(compilation),
            "compiler_options": compilation.kernel.compiler_options,
            "backend_options": request.backend_options,
            "pipeline": request.pipeline,
            "pass_options": request.pass_options,
            "architecture": _architecture(compilation),
            "toolchain": compilation_toolchain_identity(compilation),
            "versions": _compiler_versions(),
        }
    )


def request_cache_key(request):
    """Fingerprint inspectable requests before compiling; return None if uncertain.

    The key includes source dependencies, request options and compiler/toolchain
    versions. Nested code, unsupported closures or tensor histories use the
    compiled cache.
    """
    unsupported = object()

    def value_identity(value, seen=frozenset()):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(value, (tuple, list)):
            items = tuple(value_identity(item, seen=seen) for item in value)

            return unsupported if unsupported in items else items

        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            items = tuple(
                (key, value_identity(item, seen=seen))
                for key, item in sorted(value.items())
            )

            return (
                unsupported if any(item is unsupported for _, item in items) else items
            )

        if isfunction(value):
            identity = source_identity(value, seen)

            return identity if identity is not None else unsupported

        if isinstance(value, ModuleType) and value.__name__ == "ninetoothed.language":
            return ("symbolic-language", value.__name__)

        return unsupported

    def source_identity(callable_value, seen=frozenset()):
        arguments = None

        if isinstance(callable_value, partial):
            arguments = value_identity((callable_value.args, callable_value.keywords))

            if arguments is unsupported:
                return None

            callable_value = callable_value.func

        if (
            not isfunction(callable_value)
            or callable_value.__closure__
            or callable_value in seen
            or any(
                isinstance(value, CodeType)
                for value in callable_value.__code__.co_consts
            )
        ):
            return None

        seen = seen | {callable_value}
        dependencies = getclosurevars(callable_value)

        global_loads = {
            instruction.argval
            for instruction in get_instructions(callable_value)
            if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
        }

        if dependencies.nonlocals or dependencies.unbound & global_loads:
            return None

        defaults = value_identity(
            (callable_value.__defaults__, callable_value.__kwdefaults__), seen=seen
        )

        if defaults is unsupported:
            return None

        globals_identity = []

        for name, value in sorted(dependencies.globals.items()):
            identity = value_identity(value, seen=seen)

            if identity is unsupported:
                return None

            globals_identity.append((name, identity))

        filename = getsourcefile(callable_value)

        if filename is None:
            return None

        path = Path(filename)

        try:
            content = sha256(path.read_bytes()).hexdigest()
        except OSError:
            return None

        return (
            callable_value.__module__,
            callable_value.__qualname__,
            content,
            defaults,
            tuple(globals_identity),
            arguments,
        )

    def identify(request):
        arrangement = source_identity(request.arrangement)
        application = source_identity(request.application)

        if arrangement is None or application is None:
            return None

        tensors = []

        for tensor in request.tensors:
            if tensor.source is not tensor or tensor._history:
                return None

            if tensor.other is not None and not isinstance(
                tensor.other, (bool, int, float, str)
            ):
                return None

            tensors.append(
                (
                    tensor.name,
                    tuple(map(str, tensor.shape)),
                    str(tensor.dtype),
                    tensor.other,
                    tensor.constexpr,
                    tensor.value,
                    str(tensor.jagged_dim),
                )
            )

        options = {
            field.name: getattr(request, field.name)
            for field in fields(request)
            if field.name not in {"arrangement", "application", "tensors"}
        }

        return (arrangement, application, tensors, options)

    result = identify(request)

    if result is None:
        return None

    package = Path(__file__).resolve().parents[1]

    try:
        sources = sha256()

        for path in sorted(package.rglob("*.py")):
            sources.update(path.relative_to(package).as_posix().encode())
            sources.update(path.read_bytes())
    except OSError:
        return None

    return stable_digest(
        {
            "request": result,
            "compiler_sources": sources.hexdigest(),
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
    target = _target_metadata(compilation)
    architecture = {
        "machine": platform.machine(),
        "target": target,
        "cuda_arch": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    if target.get("platform") not in {None, "generic"} or target.get("compute_arch"):
        return architecture

    if backend == "cuda":
        target = str(backend_options.get("arch", "native"))
        architecture["cuda_target"] = target

        if target != "native":
            return architecture

    if backend in {"cuda", "tilelang", "triton"}:
        architecture.update(_runtime_cuda_architecture())
    return architecture


def _target_metadata(compilation) -> Mapping[str, Any]:
    target = getattr(compilation, "target", None)

    if target is not None and hasattr(target, "as_metadata"):
        return dict(target.as_metadata())

    artifact = getattr(compilation, "artifact", None)
    metadata = getattr(artifact, "metadata", {})

    if isinstance(metadata, Mapping):
        return dict(metadata.get("target", {}))
    return {}


def compilation_toolchain_identity(compilation) -> Mapping[str, Any]:
    """Return external compiler identity that affects emitted binaries."""
    backend = compilation.artifact.backend.value

    if backend == "cuda":
        return {"cuda": cuda_compiler_identity()}
    return {}


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
    "compilation_toolchain_identity",
    "read_manifest",
    "request_cache_key",
    "stable_digest",
    "write_manifest",
    "write_source",
]
