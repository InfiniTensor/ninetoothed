import pickle
import subprocess
import sys

import pytest
import torch

import ninetoothed
from ninetoothed import Tensor
from ninetoothed.compiler import (
    DEFAULT_COMPILER,
    CompileRequest,
    load_built_artifact,
)
from tests.utils import get_available_devices


def _arrangement(input, other, output):
    return tuple(tensor.tile((64,)) for tensor in (input, other, output))


def _application(input, other, output):
    output = input + other  # noqa: F841


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("backend", ("triton", "cuda", "tilelang"))
def test_aot_built_artifact_can_be_reloaded(backend, device, tmp_path):
    tensors = tuple(Tensor(shape=(257,), dtype=ninetoothed.float32) for _ in range(3))
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=tensors,
            backend=backend,
            kernel_name=f"reload_{backend}",
            max_num_configs=1,
        )
    )
    handle = DEFAULT_COMPILER.materialize(
        compilation,
        output_dir=tmp_path,
        mode="aot",
    )
    input = torch.randn(257, device=device)
    other = torch.randn_like(input)
    output = torch.empty_like(input)
    handle(input, other, output)

    reloaded = load_built_artifact(handle._built_artifact)
    reloaded_output = torch.empty_like(input)
    reloaded(input, other, reloaded_output)

    assert handle._built_artifact.binary_path is not None
    assert torch.allclose(reloaded_output, input + other)

    built_path = tmp_path / "built.pkl"
    built_path.write_bytes(pickle.dumps(handle._built_artifact))
    subprocess.run(
        [
            sys.executable,
            "-c",
            _RELOAD_IN_SUBPROCESS,
            str(built_path),
            str(device),
        ],
        check=True,
    )


_RELOAD_IN_SUBPROCESS = """
import pickle
import sys

import torch

from ninetoothed.compiler import load_built_artifact

built = pickle.loads(open(sys.argv[1], "rb").read())
launch = load_built_artifact(built)
input = torch.randn(257, device=sys.argv[2])
other = torch.randn_like(input)
output = torch.empty_like(input)
launch(input, other, output)
torch.cuda.synchronize()
if not torch.allclose(output, input + other):
    raise SystemExit("reloaded kernel produced an incorrect result")
"""
