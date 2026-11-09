import functools
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from ninetoothed import Tensor
from ninetoothed.backends.emitters.analysis import walk_ops
from ninetoothed.backends.rvne_toolchain import find_rvne_toolchain
from ninetoothed.compiler import CompileRequest, compile_kernel
from ninetoothed.compiler.runtime import materialize
from tests import test_attention as attention


@pytest.fixture(scope="module")
def rvne_sdk():
    if not os.environ.get("NINETOOTHED_RVNE_TOOLCHAIN"):
        pytest.skip("RVNE SDK is not configured")

    return find_rvne_toolchain()


@pytest.mark.parametrize("is_causal", (False, True), ids=("noncausal", "causal"))
@pytest.mark.parametrize(
    "seq_len,block_m,block_n,head_dim",
    ((1, 2, 4, 4), (5, 2, 4, 4), (7, 4, 2, 4), (5, 2, 4, 64)),
    ids=("seq1-tile2x4", "seq5-tile2x4", "seq7-tile4x2", "seq5-tile2x4-head64"),
)
def test_rvne_attention_qemu(
    tmp_path, rvne_sdk, is_causal, seq_len, block_m, block_n, head_dim
):
    shape = (2, 2, seq_len, head_dim)
    rng = np.random.default_rng(107 + seq_len)
    q, k, v = tuple(rng.standard_normal(shape).astype(np.float32) for _ in range(3))
    out = np.full(shape, np.nan, dtype=np.float32)
    q_spec, k_spec, v_spec, o_spec = tuple(
        Tensor(shape=shape, dtype="float32") for _ in range(4)
    )
    causal_spec = Tensor(0, dtype="bool", constexpr=True, value=is_causal)
    compilation = compile_kernel(
        CompileRequest(
            arrangement=functools.partial(
                attention.arrangement,
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
            ),
            application=attention.application,
            tensors=(q_spec, k_spec, v_spec, causal_spec, o_spec),
            backend="rvne",
            caller="numpy",
            backend_options={"toolchain_root": str(rvne_sdk.root)},
        )
    )
    operations = tuple(
        operation
        for block in compilation.kernel.ssa.blocks
        for operation in walk_ops(block.operations)
    )
    matrix_results = tuple(
        result.type.shape
        for operation in operations
        if operation.opcode in {"linalg.dot", "linalg.matmul"}
        for result in operation.results
    )
    assert matrix_results == (
        (str(block_m), str(block_n)),
        (str(block_m), str(shape[-1])),
    )
    assert any(operation.opcode == "scf.for" for operation in operations)
    expected = functional.scaled_dot_product_attention(
        torch.from_numpy(q),
        torch.from_numpy(k),
        torch.from_numpy(v),
        is_causal=is_causal,
        dropout_p=0.0,
        scale=1.0,
    ).numpy()
    handle = materialize(compilation, output_dir=tmp_path, mode="aot")
    built = handle._built_artifact
    assert Path(built.source_path).is_file()
    executable = Path(built.binary_path)
    assert executable.is_file()
    header = executable.read_bytes()[:20]
    assert header[:6] == b"\x7fELF\x02\x01"
    assert int.from_bytes(header[18:20], "little") == 243
    assert handle(q, k, v, is_causal, out) is out
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, expected, rtol=2e-5, atol=2e-5)
