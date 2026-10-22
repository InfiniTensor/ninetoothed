"""Full-row ownership requires complete maps and explicit rank-drop broadcasts."""

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest


def _whole(x, out):
    return x.tile(tuple(x.shape)), out.tile(tuple(out.shape))


def _softmax(x, out):
    z = ntl.exp(x - ntl.max(x, axis=1)[:, None])
    out = (z / ntl.sum(z, axis=1)[:, None]).to(out.dtype)  # noqa: F841


def test_native_full_row_hook_uses_typed_output_and_records_derived_ownership():
    output_dtype = "float16"
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_whole,
            application=_softmax,
            tensors=(
                Tensor(shape=(3, 513), dtype="float32", other=0),
                Tensor(shape=(3, 513), dtype=output_dtype, other=0),
            ),
            backend="bangc",
            backend_options={},
            max_num_configs=1,
        )
    )
    proof = compilation.artifact.metadata["bangc_full_row_tiling"]
    assert proof["route"] == "target-derived-ssa"
    assert proof["output_tile"] == (1, 513)
    assert proof["nram_bytes"] <= 192 * 1024
    source = compilation.artifact.primary_source
    assert "nt_dag_buf_" in source
    spelling = {"float32": "float", "float16": "half", "int32": "int32_t"}[output_dtype]
    assert f"sizeof({spelling}), NRAM2GDRAM" in source
    launch = source.split('extern "C"')[1]
    assert " ? " in launch and "nt_chunk" not in launch
