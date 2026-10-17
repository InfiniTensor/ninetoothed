"""Basic opt-in singleton-dot selection and hardware-bound candidate coverage."""

import sys
from dataclasses import replace
from functools import partial, wraps

import pytest
import torch

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import (
    DEFAULT_COMPILER,
    CompileRequest,
    autotune_singleton_dot,
)
from ninetoothed.compiler.cache import stable_digest
from ninetoothed.compiler.passes import Pipeline, default_spec
from ninetoothed.compiler.singleton_dot_autotune import (
    _legal_warps,
    _model_features,
    _next_expansion,
    _quick_selection_key,
    _with_pipeline,
)


def _arrangement(a, b, out):
    return a.tile((-1, -1)), b.tile((-1, -1)), out.tile((-1, -1))


def _application(a, b, out):
    out = ntl.dot(a, b)  # noqa: F841


_GLOBAL_TILE = 1


def _configured_arrangement(a, b, out):
    return a.tile((-1, -1)), b.tile((-1, -1)), out.tile((1, _GLOBAL_TILE))


def _batched_arrangement(a, b, out, *, m):
    arranged = tuple(
        tensor.tile(shape)
        for tensor, shape in zip((a, b, out), ((1, m, 32), (1, 32, 16), (1, m, 16)))
    )

    for tensor in arranged:
        tensor.dtype = tensor.dtype.squeeze(0)

    return arranged


def test_singleton_dot_search_and_pipeline(monkeypatch):
    hardware = {
        "warp_size": 32,
        "max_threads_per_block": 1024,
        "max_threads_per_sm": 2048,
        "multiprocessors": 4,
        "incomplete": False,
    }
    legal = _legal_warps(hardware)
    assert legal == (1, 2, 4, 8, 16, 32)
    unknown = hardware | {"max_threads_per_block": None}
    assert _legal_warps(unknown, requested=8) == (1, 2, 4, 8)

    with pytest.raises(ValueError, match="legal warp count"):
        _legal_warps(hardware | {"max_threads_per_block": 128}, requested=8)

    assert (
        _next_expansion(
            "original",
            {
                ("original", 4): 100,
                ("original", 8): 90,
                ("a09", 4): 100,
                ("a09", 8): 101,
            },
            (8, 16, 4, 2, 1, 32),
            legal,
            uncertain=False,
        )
        == 16
    )

    measured = {("original", 4): 90, ("a09", 8): 100, ("a09", 16): 108}
    ranked = (8, 16, 4, 32, 2, 1)
    assert _next_expansion("a09", measured, ranked, legal, uncertain=True) == 4
    measured[("a09", 4)] = 107
    assert _next_expansion("a09", measured, ranked, legal, uncertain=True) == 32
    spec = default_spec("triton")
    request = CompileRequest(application=_application, pipeline=spec)
    optimized = _with_pipeline(request, enable=True)
    assert optimized.pipeline.passes.count("ssa.triton.singleton_dot") == 1
    assert _with_pipeline(optimized, enable=False).pipeline == spec

    with pytest.raises(ValueError, match="declarative pipeline"):
        _with_pipeline(replace(request, pipeline=Pipeline(())), enable=True)

    configured = replace(request, arrangement=_configured_arrangement)
    key_before = _quick_selection_key(
        configured, configured, "example", hardware, 0.03, stable_digest
    )
    monkeypatch.setattr(sys.modules[__name__], "_GLOBAL_TILE", 2)
    key_after = _quick_selection_key(
        configured, configured, "example", hardware, 0.03, stable_digest
    )
    assert key_before is not None and key_after != key_before

    args = (
        torch.empty(8, 1, 32, dtype=torch.float16),
        torch.empty(8, 32, 16, dtype=torch.float16),
        torch.empty(8, 1, 16),
    )
    batched = CompileRequest(
        arrangement=partial(_batched_arrangement, m=16),
        application=_application,
        tensors=tuple(
            Tensor(shape=arg.shape, dtype=str(arg.dtype).split(".")[-1]) for arg in args
        ),
        backend="triton",
    )
    base = DEFAULT_COMPILER.compile(batched)
    a09 = DEFAULT_COMPILER.compile(
        _with_pipeline(
            replace(batched, arrangement=partial(_batched_arrangement, m=1)),
            enable=True,
        )
    )
    features = _model_features(base, a09, args, 2, hardware, ((1, 32, 16),))
    assert features["original"]["programs"] == features["a09"]["programs"] == 8
    assert features["original"]["padding"] > features["a09"]["padding"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_singleton_dot_autotune(tmp_path, monkeypatch):
    from ninetoothed import auto_tuner
    from ninetoothed.compiler import cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(auto_tuner, "_AUTO_TUNING_CACHE_DIR", tmp_path / "auto_tuning")
    properties = torch.cuda.get_device_properties(0)

    class OlderProperties:
        def __getattr__(self, name):
            if name == "max_threads_per_block":
                raise AttributeError(name)
            return getattr(properties, name)

    # Exercise the public entry with a legal caller configuration absent from
    # the conservative search range, as with older PyTorch property bindings.
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: OlderProperties()
    )

    request = CompileRequest(
        arrangement=_arrangement,
        application=_application,
        tensors=(
            Tensor(shape=(1, 32), dtype="float16"),
            Tensor(shape=(32, 16), dtype="float16"),
            Tensor(shape=(1, 16), dtype="float32"),
        ),
        backend="triton",
        num_stages=2,
        num_warps=8,
    )
    a = torch.randn(1, 32, device="cuda", dtype=torch.float16)
    b = torch.randn(32, 16, device="cuda", dtype=torch.float16)
    out = torch.empty(1, 16, device="cuda", dtype=torch.float32)
    tuned = autotune_singleton_dot(request, request, (a, b, out))
    assert {candidate["kind"] for candidate in tuned.candidates} == {"original", "a09"}
    torch.testing.assert_close(
        out, (a.double() @ b.double()).float(), atol=5e-4, rtol=2e-5
    )

    a.normal_()
    tuned(a, b, out)
    torch.testing.assert_close(
        out, (a.double() @ b.double()).float(), atol=5e-4, rtol=2e-5
    )
    original_compile = DEFAULT_COMPILER.compile
    compiled = []

    def counted_compile(compile_request):
        compiled.append(compile_request)

        return original_compile(compile_request)

    monkeypatch.setattr(DEFAULT_COMPILER, "compile", counted_compile)
    cached = autotune_singleton_dot(request, request, (a, b, out))
    assert cached.cache_hit
    assert len(compiled) == 1
    assert cached.selected(a, b, out)["warps"] == tuned.selected(a, b, out)["warps"]

    alias_out = b.view(torch.float32).reshape(-1)[:16].view(1, 16)

    # Changing storage on the same output object must invalidate the guard.
    out.set_(alias_out)

    with pytest.raises(ValueError, match="aliased outputs"):
        cached(a, b, out)

    out.set_(torch.empty_like(out))
    cached(a, b, out)
    torch.testing.assert_close(
        out, (a.double() @ b.double()).float(), atol=5e-4, rtol=2e-5
    )

    strided = torch.empty_strided(a.shape, (64, 2), device="cuda", dtype=a.dtype)
    strided.copy_(a)
    original_materialize = DEFAULT_COMPILER.materialize
    launches = []

    def counted_materialize(compilation):
        handle = original_materialize(compilation)
        launch = handle._launch
        invoke = launch._ninetoothed_invoke_prepared
        prepare = launch._ninetoothed_prepare

        @wraps(invoke)
        def counted_invoke(prepared, call_args, call_kwargs):
            if call_args[-1] is out:
                launches.append(1)

            return invoke(prepared, call_args, call_kwargs)

        @wraps(prepare)
        def counted_prepare(*call_args, **call_kwargs):
            return replace(prepare(*call_args, **call_kwargs), invoker=counted_invoke)

        launch._ninetoothed_prepare = counted_prepare

        return handle

    monkeypatch.setattr(DEFAULT_COMPILER, "materialize", counted_materialize)
    assert tuned(strided, b, out) is out
    assert len(launches) == 1
    assert tuned(strided, b, out) is out
    assert len(launches) == 2
    torch.testing.assert_close(
        out, (strided.double() @ b.double()).float(), atol=5e-4, rtol=2e-5
    )
