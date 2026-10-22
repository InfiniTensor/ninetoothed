"""Complete-row legality and resource contracts for the Ascend DAG adapter."""

import pytest

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import lower


def arrangement(x, y, out):
    return x.tile((1, x.shape[1])), y.tile((1, y.shape[1])), out.tile((1, out.shape[1]))


def rms(x, y, out):
    out = x / ntl.sqrt(ntl.sum(x * x, axis=1) / x.shape[1] + 0.00001)  # noqa: F841


def statistics(x, y, out):
    mean = ntl.sum(x, axis=1) / x.shape[1]
    out = ntl.sum((x - mean) * (x - mean), axis=1) / x.shape[1]  # noqa: F841


def probabilities(x, y, out):
    value = ntl.exp(x - ntl.max(x, axis=1))
    out = value / ntl.sum(value, axis=1)  # noqa: F841


@pytest.mark.parametrize(
    "application,collapsed", ((rms, False), (statistics, True), (probabilities, False))
)
def test_complete_row_dag_reuses_values_and_keeps_the_indexed_fallback(
    application, collapsed
):
    cols = 136
    source = lower(
        arrangement,
        application,
        (
            Tensor(shape=(17, cols), dtype="float32", other=0),
            Tensor(shape=(17, cols), dtype="float32", other=0),
            Tensor(shape=(17, 1 if collapsed else cols), dtype="float32", other=0),
        ),
        backend="ascendc",
        kernel_name="row_dag_contract",
    ).primary_source
    assert "Shared typed row DAG" in source
    assert "ReduceSum(nt_ref1" in source
    assert "nt_launch_logical * 16" in source
    assert "& 63ULL" in source
    assert "if (nt_launch_logical != 0)" in source
    assert "nt_slice <" in source

    if application is probabilities:
        assert "ReduceMax(nt_ref1" in source
        assert "nt_inf(), CMPMODE::LE, (uint32_t)nt_nan_prefix" in source
        assert "nt_nan_bits.GetValue(nt_word) != ~0ULL" in source
        assert "= nt_nan(); break;" in source
        assert "(nt_cols) / 128 * 128" in source
        assert "nt_word * 64 < nt_nan_prefix" in source
        assert "nt_j = nt_nan_prefix; nt_j < (nt_cols)" in source
        assert "nt_tail_value != nt_tail_value" in source
