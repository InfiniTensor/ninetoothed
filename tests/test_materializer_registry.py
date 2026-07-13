import inspect

import pytest

from ninetoothed.backends.core import Target
from ninetoothed.backends.materializers import create_default_registry
from ninetoothed.backends.materializers import triton as triton_materializer
from ninetoothed.backends.materializers.base import MaterializerRegistry
from ninetoothed.backends.materializers.cuda import CudaMaterializer


def test_each_builtin_backend_implements_materializer_contract():
    registry = create_default_registry()

    for target in Target:
        materializer = registry.get(target)
        assert callable(materializer.jit_materialize)
        assert callable(materializer.aot_build)
        assert callable(materializer.load_built_artifact)


def test_materializer_registry_rejects_duplicate_without_replace():
    registry = MaterializerRegistry()
    registry.register(CudaMaterializer())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(CudaMaterializer())

    registry.register(CudaMaterializer(), replace=True)


def test_triton_materializer_uses_shared_toolchain_boundary():
    source = inspect.getsource(triton_materializer)
    assert "ninetoothed.backends.toolchain import find_nvcc" in source
    assert "ninetoothed.backends.materializers.cuda" not in source
