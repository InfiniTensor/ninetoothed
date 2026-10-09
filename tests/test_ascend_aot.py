import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import ninetoothed
from ninetoothed import Tensor
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest, load_built_artifact
from tests.test_ascend_runtime import _add, _arrangement, _fused


@pytest.mark.parametrize("application", (_add, _fused))
@pytest.mark.parametrize("dtype", ("float32", "float16"))
def test_aot_build_reload_without_compiler(application, dtype, tmp_path):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=application,
            tensors=tuple(
                Tensor(1, dtype=getattr(ninetoothed, dtype)) for _ in range(3)
            ),
            backend="ascend",
            platform="ascend-910b4",
        )
    )
    handle = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    built = handle._built_artifact
    assert Path(built.binary_path).stat().st_size > 0
    relocated = tmp_path / "relocated"
    shutil.copytree(Path(built.binary_path).parent, relocated)
    built = replace(built, binary_path=str(relocated / "kernel.bin"))
    for launch in (handle, load_built_artifact(built)):
        for size in (1, 513, 98432):
            x = torch.randn(size, device="npu", dtype=getattr(torch, dtype))
            y = torch.randn_like(x)
            out = torch.empty_like(x)
            launch(x, y, out)
            torch.npu.synchronize()
            expected = x + y if application is _add else (x + y) * (x - y)
            torch.testing.assert_close(out, expected)
    path = tmp_path / "built.pkl"
    path.write_bytes(pickle.dumps(built))
    env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / "empty-cache"))
    subprocess.run(
        [sys.executable, "-c", _RELOAD, str(path), application.__name__, dtype],
        check=True,
        env=env,
    )
    wrong = torch.ones(513, device="npu", dtype=torch.int32)
    with pytest.raises((TypeError, ValueError), match="dtype"):
        load_built_artifact(built)(wrong, wrong, wrong)
    binary = Path(built.binary_path)
    binary.write_bytes(binary.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_built_artifact(built)


def test_aot_rejects_unspecified_dtypes(tmp_path):
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_add,
            tensors=tuple(Tensor(1) for _ in range(3)),
            backend="ascend",
            platform="ascend-910b4",
        )
    )
    with pytest.raises(ValueError, match="explicit tensor dtypes"):
        DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")


_RELOAD = """
import pickle, sys, subprocess
import torch, torch_npu, triton
from ninetoothed.compiler import load_built_artifact
def forbidden(*args, **kwargs):
    raise AssertionError('AOT reload must not compile')
triton.compile = forbidden
subprocess.run = forbidden
subprocess.Popen = forbidden
built = pickle.loads(open(sys.argv[1], 'rb').read())
launch = load_built_artifact(built)
x = torch.randn(513, device='npu', dtype=getattr(torch, sys.argv[3]))
y = torch.randn_like(x)
out = torch.empty_like(x)
launch(x, y, out)
torch.npu.synchronize()
expected = x + y if sys.argv[2] == '_add' else (x + y) * (x - y)
torch.testing.assert_close(out, expected)
"""
