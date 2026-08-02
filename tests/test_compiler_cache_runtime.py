from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ninetoothed.backends.materializers.cuda import _cuda_wrapper
from ninetoothed.compiler.cache import (
    compilation_cache_key,
    stable_digest,
    write_source,
)
from ninetoothed.compiler.runtime import _public_values, _runtime_wrapper
from ninetoothed.ir import LaunchABI, TensorSpec


class _Tensor:
    def __init__(self, shape, dtype="float32", device_type="cuda", device_index=0):
        self.shape = shape
        self.dtype = dtype
        self.device = SimpleNamespace(type=device_type, index=device_index)


def _abi():
    return LaunchABI(public_args=("x", "out"), outputs=("out",))


def _specs():
    return (
        TensorSpec(name="x", ndim=2, shape=("m", "n"), dtype="float32"),
        TensorSpec(name="out", ndim=2, shape=("m", "n"), dtype="float32"),
    )


def _compilation(backend, backend_options=None):
    return SimpleNamespace(
        request=SimpleNamespace(
            backend_options=backend_options or {},
            pipeline=None,
            pass_options=None,
        ),
        artifact=SimpleNamespace(
            backend=SimpleNamespace(value=backend),
            sources={"kernel": "source"},
        ),
        kernel=SimpleNamespace(ssa=(), compiler_options={}),
        launch_plan=(),
    )


def test_runtime_binding_rejects_unknown_duplicate_and_missing_arguments():
    x = _Tensor((2, 3))
    out = _Tensor((2, 3))

    with pytest.raises(TypeError, match="Unknown kernel arguments"):
        _public_values(_abi(), (x, out), {"extra": 1}, specs=_specs())

    with pytest.raises(TypeError, match="passed twice"):
        _public_values(_abi(), (x,), {"x": x, "out": out}, specs=_specs())

    with pytest.raises(TypeError, match="Missing kernel arguments"):
        _public_values(_abi(), (x,), {}, specs=_specs())


@pytest.mark.parametrize(
    "value, message",
    (
        (_Tensor((6,)), "rank 1; expected 2"),
        (_Tensor((2, 3), dtype="float16"), "dtype float16; expected float32"),
        (_Tensor((2, 3), device_type="cpu"), "must be on a CUDA device"),
    ),
)
def test_runtime_binding_validates_tensor_contract(value, message):
    with pytest.raises(TypeError, match=message):
        _public_values(
            _abi(),
            (value, _Tensor((2, 3))),
            {},
            specs=_specs(),
        )


def test_runtime_binding_validates_static_dimensions_of_dynamic_shape():
    spec = TensorSpec(
        name="x",
        ndim=2,
        shape=("rows", "127"),
        dtype="float32",
        attrs={"source_ndim": 2, "source_shape": ("rows", "127")},
    )

    with pytest.raises(TypeError, match="expected dimension 1 to be 127"):
        _public_values(
            LaunchABI(public_args=("x",)),
            (_Tensor((3, 64)),),
            {},
            specs=(spec,),
        )


def test_runtime_wrappers_skip_empty_launches():
    abi = LaunchABI(public_args=("out",), outputs=("out",))
    output = SimpleNamespace(numel=lambda: 0)
    calls = []

    def launch(*args):
        calls.append(args)

    assert _runtime_wrapper(launch, abi)(output) is output
    assert _cuda_wrapper(launch, abi, ())(output) is output
    assert not calls


def test_content_digest_is_stable_for_a_b_a_sources():
    a1 = stable_digest({"name": "same", "source": "A"})
    b = stable_digest({"name": "same", "source": "B"})
    a2 = stable_digest({"name": "same", "source": "A"})
    assert a1 == a2
    assert a1 != b


@pytest.mark.parametrize("backend", ("cuda", "tilelang", "triton"))
def test_native_gpu_capability_is_part_of_compilation_cache_key(backend, monkeypatch):
    import ninetoothed.compiler.cache as cache

    monkeypatch.setattr(
        cache,
        "_runtime_cuda_architecture",
        lambda: {
            "cuda_current_capability": "sm_80",
            "cuda_visible_capabilities": ("sm_80",),
        },
    )
    first = compilation_cache_key(_compilation(backend, {"arch": "native"}))
    monkeypatch.setattr(
        cache,
        "_runtime_cuda_architecture",
        lambda: {
            "cuda_current_capability": "sm_90",
            "cuda_visible_capabilities": ("sm_90",),
        },
    )
    second = compilation_cache_key(_compilation(backend, {"arch": "native"}))

    assert first != second


def test_explicit_cuda_arch_does_not_query_runtime_device(monkeypatch):
    import ninetoothed.compiler.cache as cache

    def unexpected_query():
        raise AssertionError("Explicit CUDA targets must not query a runtime device.")

    monkeypatch.setattr(cache, "_runtime_cuda_architecture", unexpected_query)
    compilation_cache_key(_compilation("cuda", {"arch": "sm_90"}))


def test_concurrent_source_writes_are_atomic(tmp_path, monkeypatch):
    import ninetoothed.compiler.cache as cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    source = "def kernel():\n    return 1\n"

    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = tuple(
            executor.map(
                lambda _: write_source("kernel", source, "py"),
                range(32),
            )
        )

    assert len(set(paths)) == 1
    assert paths[0].read_text(encoding="utf-8") == source
