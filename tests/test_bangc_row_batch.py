"""Compile a complete row graph through the default BangC batch route."""

import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest


def _rows(x, y, out):
    return tuple(value.tile((1, value.shape[1])) for value in (x, y, out))


def _softmax(x, y, out):
    z = ntl.exp(x - ntl.max(x, axis=1))
    out = z / ntl.sum(z, axis=1)  # noqa: F841


def test_actual_batch_hook_retains_original_grid_and_records_resource_proof():
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_rows,
            application=_softmax,
            tensors=tuple(
                nt.Tensor(shape=(17, 1024), dtype=nt.float32, other=0) for _ in range(3)
            ),
            backend="bangc",
            backend_options={},
            max_num_configs=1,
        )
    )
    source = compilation.artifact.primary_source
    assert "__bang_sumpool" in source
    assert " ? (3) : " in source.split('extern "C"')[1]
    assert "nt_batch_base" in source
    metadata = compilation.artifact.metadata
    assert metadata["bangc_row_batch"]["batch_rows"] == 8
    assert metadata["bangc_row_batch"]["dot_accumulation"] == "unchanged"
