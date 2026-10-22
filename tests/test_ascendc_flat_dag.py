"""Flat DAG packets preserve the original pointwise domain and effects."""

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import lower


def flat(x, y, out, auxiliary):
    return tuple(value.tile((128,)) for value in (x, y, out, auxiliary))


def deep(x, y, out, auxiliary):
    value = x * 0.5 + y * 0.25
    value = ntl.exp(value) + 1
    value = ntl.log(value) * 0.75
    value = ntl.sqrt(value * value + 0.0001)
    out = (value + x * 0.125) * 0.5  # noqa: F841
    auxiliary = value - y * 0.5  # noqa: F841


def test_flat_dag_stages_multiple_outputs_with_a_complete_packet_proof():
    extent, application = (16384, deep)
    source = lower(
        flat,
        application,
        tuple((Tensor(shape=(extent,), dtype="float32", other=0) for _ in range(4))),
        backend="ascendc",
        kernel_name="flat_dag",
    ).primary_source
    assert "Shared typed flat DAG" in source
    assert "nt_pipe.InitBuffer(nt_dag_buf" in source
    assert "& 63ULL" in source
    assert "if (nt_launch_logical != 0)" in source
    assert "for (int64_t index = 0; index <" in source
