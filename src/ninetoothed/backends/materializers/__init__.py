"""Built-in backend materializer registry."""

from ninetoothed.backends.core import Target
from ninetoothed.backends.materializers.base import MaterializerRegistry


def create_default_registry() -> MaterializerRegistry:
    from ninetoothed.backends.materializers.bangc import BangCMaterializer
    from ninetoothed.backends.materializers.cuda import CudaMaterializer
    from ninetoothed.backends.materializers.tilelang import TileLangMaterializer
    from ninetoothed.backends.materializers.triton import TritonMaterializer

    registry = MaterializerRegistry()
    registry.register(TritonMaterializer())
    registry.register(CudaMaterializer())
    registry.register(TileLangMaterializer())
    registry.register(BangCMaterializer())

    return registry


_DEFAULT_REGISTRY: MaterializerRegistry | None = None


def materializer_for(target: Target | str):
    global _DEFAULT_REGISTRY

    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = create_default_registry()

    return _DEFAULT_REGISTRY.get(target)


__all__ = ["create_default_registry", "materializer_for"]
