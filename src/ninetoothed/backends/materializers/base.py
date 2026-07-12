"""Backend materialization contracts."""

from abc import ABC, abstractmethod
from pathlib import Path

from ninetoothed.backends.core import BuiltArtifact, Target, backend_id_for


class Materializer(ABC):
    """Build and reload executable artifacts for one backend."""

    target: Target | str

    @abstractmethod
    def jit_materialize(self, compilation, *, output_dir: str | Path | None = None):
        """Create a callable for interactive JIT execution."""

    @abstractmethod
    def aot_build(self, compilation, *, output_dir: str | Path):
        """Build a reloadable binary and return a callable handle."""

    @abstractmethod
    def load_built_artifact(self, built: BuiltArtifact):
        """Load an existing binary without recompiling it."""


class MaterializerRegistry:
    def __init__(self):
        self._materializers: dict[str, Materializer] = {}

    def register(self, materializer: Materializer, *, replace: bool = False) -> None:
        backend_id = backend_id_for(materializer.target)

        if backend_id in self._materializers and not replace:
            raise ValueError(f"Materializer `{backend_id}` is already registered.")

        self._materializers[backend_id] = materializer

    def get(self, target: Target | str) -> Materializer:
        backend_id = backend_id_for(target)

        try:
            return self._materializers[backend_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._materializers))
            raise ValueError(
                f"No materializer for `{backend_id}`. Available: {available}."
            ) from exc


__all__ = ["Materializer", "MaterializerRegistry"]
