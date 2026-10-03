"""Real PyTorch CPU adapter checks; unavailable dependencies are not passes."""

import os
import subprocess
import sys

import numpy as np
import pytest

from ninetoothed import Tensor, interpret
from ninetoothed.interpreter import InterpretationError, interpret_program
from ninetoothed.ir import ssa


@pytest.fixture(scope="module")
def torch():
    try:
        import torch as module
    except ImportError as error:
        pytest.fail(f"Torch adapter validation UNVERIFIED: {error}.", pytrace=False)
    return module


def _arrangement(x, y, out):
    return tuple(value.tile((4,)) for value in (x, y, out))


def _add(x, y, out):
    out = x + y  # noqa: F841


def _kernel(dtype="float32"):
    return interpret(
        _arrangement,
        _add,
        tuple(Tensor(1, name=name, dtype=dtype) for name in ("x", "y", "out")),
    )


@pytest.mark.parametrize("dtype_name", ("float32", "int32"))
def test_torch_cpu_output_keeps_identity_and_updates_shared_storage(torch, dtype_name):
    dtype = getattr(torch, dtype_name)
    x = torch.arange(11, dtype=dtype)
    y = torch.ones(11, dtype=dtype)
    out = torch.empty_like(x)
    numpy_view = out.numpy()
    result = _kernel(dtype_name)(x, y, out)
    assert result.outputs["out"] is out
    np.testing.assert_array_equal(numpy_view, (x + y).numpy())
    torch.testing.assert_close(out, x + y, rtol=0, atol=0)


def test_input_adapter_shares_cpu_storage_before_numeric_load(torch):
    tensor_type = ssa.Type(kind="tensor", shape=("3",), dtype="float32")
    x = ssa.Value(name="x", type=tensor_type)
    out = ssa.Value(name="out", type=tensor_type)
    one = ssa.Value(name="%one", type=ssa.Type(kind="scalar", dtype="float32"))
    result = ssa.Value(name="%sum", type=tensor_type)
    program = ssa.Program(
        kind="zero_copy_adapter",
        inputs=(x, out),
        outputs=(out,),
        blocks=(
            ssa.Block(
                operations=(
                    ssa.Operation(
                        opcode="arith.constant", results=(one,), attrs={"value": 1}
                    ),
                    ssa.Operation(
                        opcode="arith.add", operands=("x", "%one"), results=(result,)
                    ),
                    ssa.Operation(opcode="mem.store", operands=("%sum", "out")),
                )
            ),
        ),
    )
    source = torch.tensor([1, 2, 3], dtype=torch.float32)
    output = torch.empty_like(source)

    def modify_after_adapter(event):
        if event.opcode == "arith.constant":
            source[0] = 42

    actual = interpret_program(
        program, {"x": source, "out": output}, callback=modify_after_adapter
    )
    assert actual.outputs["out"] is output
    torch.testing.assert_close(
        output, torch.tensor([43, 3, 4], dtype=torch.float32), rtol=0, atol=0
    )


def test_noncontiguous_torch_cpu_views_preserve_strides_and_guard_storage(torch):
    x = torch.arange(33, dtype=torch.float32)[1::3]
    y = torch.ones(11, dtype=torch.float32)
    backing = torch.full((35,), -731, dtype=torch.float32)
    out = backing[1:34:3]
    assert not x.is_contiguous() and not out.is_contiguous()
    result = _kernel()(x, y, out)
    assert result.outputs["out"] is out
    torch.testing.assert_close(out, x + y, rtol=0, atol=0)
    unchanged = torch.ones(35, dtype=torch.bool)
    unchanged[1:34:3] = False
    torch.testing.assert_close(
        backing[unchanged], torch.full_like(backing[unchanged], -731), rtol=0, atol=0
    )


@pytest.mark.parametrize("grad_binding", ("x", "out"))
def test_requires_grad_torch_buffers_are_rejected(torch, grad_binding):
    inputs = {name: torch.ones(8, dtype=torch.float32) for name in ("x", "y", "out")}
    inputs[grad_binding].requires_grad_(True)

    with pytest.raises(
        (InterpretationError, TypeError), match="requires_grad|gradient"
    ):
        _kernel()(**inputs)


def test_sparse_torch_input_is_rejected(torch):
    sparse = torch.sparse_coo_tensor([[0, 4]], [1.0, 2.0], size=(8,))

    with pytest.raises((InterpretationError, TypeError), match="strided|sparse|layout"):
        _kernel()(sparse, torch.ones(8), torch.empty(8))


def test_cuda_tensor_is_rejected_by_cpu_interpreter(torch):
    if not torch.cuda.is_available():
        pytest.fail(
            "CUDA-input rejection validation UNVERIFIED: no CUDA GPU.", pytrace=False
        )

    with pytest.raises((InterpretationError, TypeError), match="CPU|cpu|CUDA|cuda"):
        _kernel()(torch.ones(8, device="cuda"), torch.ones(8), torch.empty(8))


def test_torch_dtype_must_match_declared_ssa_dtype(torch):
    with pytest.raises(InterpretationError, match="dtype"):
        _kernel()(torch.ones(8, dtype=torch.int32), torch.ones(8), torch.empty(8))


def test_numpy_only_execution_does_not_import_torch():
    code = """
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise AssertionError('NumPy CPU path imported torch')
sys.meta_path.insert(0, NoTorch())
import numpy as np
from tests.test_interpreter_torch import _kernel
x = np.arange(11, dtype=np.float32)
out = np.empty_like(x)
_kernel()(x, x, out)
np.testing.assert_array_equal(out, x + x)
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
