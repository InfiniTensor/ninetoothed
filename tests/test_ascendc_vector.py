"""AscendC native vectors consume the same proved core as BangC."""

from dataclasses import replace

import pytest

from ninetoothed.backends.emitters import native_vector
from ninetoothed.compiler import DEFAULT_COMPILER
from tests._vector_backend_cases import check_vector_numerics, vector_request


def request(n=1031, dtype="float32", candidate="vector-1024"):
    return vector_request("ascendc", n, dtype, candidate)


def test_vector_schedule_and_storage(monkeypatch):
    contexts = []
    original = native_vector.prove_flat_domain

    def capture(context):
        contexts.append(context)

        return original(context)

    monkeypatch.setattr(native_vector, "prove_flat_domain", capture)
    compilation = DEFAULT_COMPILER.compile(request())
    source = compilation.artifact.primary_source
    assert "::AscendC::Add(" in source and "::AscendC::Mul(" in source
    assert "ReleaseEventID" in source and "DataCopyPad" in source
    assert "::AscendC::GetSubBlockIdx() == 0" in source
    assert (
        compilation.artifact.metadata["ascendc_vector"]["storage_bytes"]
        <= 192 * 1024 - 256
    )
    assert (
        "nt_vec_pipe"
        not in DEFAULT_COMPILER.compile(request(candidate=None)).artifact.primary_source
    )
    assert (
        "CAST_RINT"
        in DEFAULT_COMPILER.compile(request(dtype="float16")).artifact.primary_source
    )
    assert (
        "nt_vec_pipe"
        in DEFAULT_COMPILER.compile(
            request(candidate="vector-4096")
        ).artifact.primary_source
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

    with pytest.raises(ValueError):
        DEFAULT_COMPILER.compile(request(dtype="int32"))


def test_vector_native_tail_stream_reload_and_fallback(tmp_path):
    import torch

    pytest.importorskip("torch_npu")

    if not torch.npu.is_available():
        pytest.skip("NPU device required")

    check_vector_numerics(tmp_path, backend="ascendc", device="npu")
