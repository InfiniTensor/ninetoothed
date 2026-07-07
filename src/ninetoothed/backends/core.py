"""Backend registry and artifact contracts for NineToothed."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from ninetoothed.ir import Kernel


class Target(str, Enum):
    TRITON = "triton"
    TILELANG = "tilelang"
    CUDA = "cuda"
    TVM = "tvm"


_CANONICAL_BACKEND_NAMES = {
    None: Target.TRITON,
    "triton": Target.TRITON,
    "tilelang": Target.TILELANG,
    "cuda": Target.CUDA,
    "tvm": Target.TVM,
}


@dataclass(frozen=True, kw_only=True)
class Options:
    """Options passed from public APIs to a backend lowerer."""

    name: Target = Target.TRITON
    caller: str | None = None
    emit_only: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class Capability:
    """Human-readable status for a backend implementation."""

    name: Target
    emits_source: bool
    can_execute: bool
    requires_external_compiler: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Artifact:
    """The output of lowering a :class:`Kernel` to a backend."""

    backend: Target
    kernel_name: str
    language: str
    sources: Mapping[str, str]
    entrypoint: str | None = None
    executable: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def primary_source_name(self) -> str:
        return next(iter(self.sources))

    @property
    def primary_source(self) -> str:
        return self.sources[self.primary_source_name]

    def write_to(self, output_dir: str | Path) -> tuple[Path, ...]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        paths = []

        for name, content in self.sources.items():
            path = output_path / name
            path.write_text(content, encoding="utf-8")
            paths.append(path)

        return tuple(paths)


class Backend:
    """Base class for source and executable backend emitters."""

    name: Target
    capability: Capability

    def emit(self, kernel: Kernel, options: Options | None = None) -> Artifact:
        raise NotImplementedError


class Registry:
    """Small explicit registry to avoid import-time backend guessing."""

    def __init__(self):
        self._backends: MutableMapping[Target, Backend] = {}

    def register(self, backend: Backend) -> None:
        self._backends[backend.name] = backend

    def get(self, name: Target | str | None) -> Backend:
        normalized = normalize_target(name)

        try:
            return self._backends[normalized]
        except KeyError as exc:
            available = ", ".join(sorted(backend.value for backend in self._backends))
            raise ValueError(
                f"Unsupported backend `{normalized.value}`. Available backends: {available}."
            ) from exc

    def capabilities(self) -> tuple[Capability, ...]:
        return tuple(backend.capability for backend in self._backends.values())


def normalize_target(name: Target | str | None) -> Target:
    if isinstance(name, Target):
        return name

    key = None if name is None else str(name).lower()

    try:
        return _CANONICAL_BACKEND_NAMES[key]
    except KeyError as exc:
        supported = ", ".join(
            sorted(
                alias for alias in _CANONICAL_BACKEND_NAMES if isinstance(alias, str)
            )
        )
        raise ValueError(
            f"Unsupported backend `{name}`. Supported backends: {supported}."
        ) from exc


def normalize_options(
    backend: Target | str | None = None,
    *,
    caller: str | None = None,
    emit_only: bool = True,
    **extra: Any,
) -> Options:
    return Options(
        name=normalize_target(backend),
        caller=caller,
        emit_only=emit_only,
        extra=extra,
    )
