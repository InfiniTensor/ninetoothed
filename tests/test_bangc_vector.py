"""Pure vector fusion, its explicit boundary and native execution."""

from dataclasses import replace

import pytest

from ninetoothed.compiler import DEFAULT_COMPILER
from tests._vector_backend_cases import check_vector_numerics, vector_request


def request(n=1031, dtype="float32", candidate="vector-1024"):
    return vector_request("bangc", n, dtype, candidate)


def test_vector_schedule_and_typed_storage(monkeypatch):
    from ninetoothed.backends.emitters import native_vector

    contexts = []
    original = native_vector.prove_flat_domain

    def capture(context):
        contexts.append(context)

        return original(context)

    monkeypatch.setattr(native_vector, "prove_flat_domain", capture)
    compilation = DEFAULT_COMPILER.compile(request())
    source = compilation.artifact.primary_source
    assert "__bang_add(" in source and "__bang_mul(" in source
    assert "__bang_mul_scalar(" in source
    assert "GDRAM2NRAM" in source and "NRAM2GDRAM" in source
    assert compilation.artifact.metadata["bangc_vector"]["storage_bytes"] <= 384 * 1024
    assert (
        "__bang_add("
        not in DEFAULT_COMPILER.compile(request(candidate=None)).artifact.primary_source
    )
    assert (
        "nt_vec_start"
        in DEFAULT_COMPILER.compile(
            request(candidate="vector-4096")
        ).artifact.primary_source
    )

    assert (
        "__bang_float2half_rn"
        in DEFAULT_COMPILER.compile(request(dtype="float16")).artifact.primary_source
    )
    context = contexts[0]
    changed = replace(
        context.value_types["x"],
        attrs={**context.value_types["x"].attrs, "dtype_level": 1},
    )
    assert (
        original(replace(context, value_types={**context.value_types, "x": changed}))
        is None
    )
    assert (
        original(
            replace(
                context,
                value_types={
                    **context.value_types,
                    "x": replace(context.value_types["x"], shape=("1",)),
                },
            )
        )
        is None
    )

    with pytest.raises(ValueError):
        DEFAULT_COMPILER.compile(request(dtype="int32"))


def test_vector_native_tail_reload_and_fallback(tmp_path):
    import torch

    pytest.importorskip("torch_mlu")

    if not torch.mlu.is_available():
        pytest.skip("MLU device required")

    check_vector_numerics(tmp_path, backend="bangc", device="mlu")
