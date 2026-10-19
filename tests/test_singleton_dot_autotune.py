"""Exercise singleton-dot selection, reuse, and tensor-signature changes."""

import pytest
import torch

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import CompileRequest, autotune_singleton_dot


def _arrangement(a, b, out):
    return a.tile((-1, -1)), b.tile((-1, -1)), out.tile((-1, -1))


def _application(a, b, out):
    out = ntl.dot(a, b)  # noqa: F841


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_singleton_dot_autotune(tmp_path, monkeypatch):
    from ninetoothed.compiler import cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
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
    assert {candidate["kind"] for candidate in tuned.candidates} == {
        "original",
        "singleton_reduction",
    }
    torch.testing.assert_close(
        out, (a.double() @ b.double()).float(), atol=5e-4, rtol=2e-5
    )

    a.normal_()
    tuned(a, b, out)
    torch.testing.assert_close(
        out, (a.double() @ b.double()).float(), atol=5e-4, rtol=2e-5
    )
    cached = autotune_singleton_dot(request, request, (a, b, out))
    assert cached.cache_hit
    torch.testing.assert_close(
        out, (a.double() @ b.double()).float(), atol=5e-4, rtol=2e-5
    )

    strided = torch.empty_strided(a.shape, (64, 2), device="cuda", dtype=a.dtype)
    strided.copy_(a)
    assert tuned(strided, b, out) is out
    assert tuned.selected(strided, b, out) is not None
    torch.testing.assert_close(
        out, (strided.double() @ b.double()).float(), atol=5e-4, rtol=2e-5
    )
