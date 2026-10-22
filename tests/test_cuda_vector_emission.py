"""Check vector source legality and exact scalar-to-component substitution."""

from ninetoothed import Tensor, float32
from ninetoothed.compiler import lower


def _flat_arrangement(x, y, out):
    return x.tile((256,)), y.tile((256,)), out.tile((256,))


def _flat_application(x, y, out):
    out = x + y  # noqa: F841


def test_public_float32_add_reaches_the_vector_renderer():
    source = lower(
        _flat_arrangement,
        _flat_application,
        tuple(Tensor(1, dtype=float32) for _ in range(3)),
        backend="cuda",
    ).primary_source

    assert "float4 nt4_out_0;" in source
    assert "alignof(float4)" in source
    assert "static_cast<float>" in source
