"""End-to-end tests for the BANG (Cambricon MLU) backend materializer."""

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


def _bang_compiler_available():
    try:
        from ninetoothed.backends.toolchain import find_cncc
    except RuntimeError:
        return False

    try:
        find_cncc()
    except RuntimeError:
        return False

    return True


def _arrangement(input, other, output):
    return tuple(tensor.tile((64,)) for tensor in (input, other, output))


def _application(input, other, output):
    output = input + other  # noqa: F841


_BANG_COMPILER_AVAILABLE = _bang_compiler_available()
_HAS_MLU = any(str(device) == "mlu" for device in get_available_devices())


def test_bang_source_is_emitted():
    """Source emission needs no device and no compiler."""
    tensors = tuple(Tensor(shape=(257,), dtype=ninetoothed.float32) for _ in range(3))
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=tensors,
            backend="bang",
            kernel_name="bang_emit",
            max_num_configs=1,
        )
    )

    source = compilation.artifact.primary_source

    assert "__mlu_global__" in source
    assert "cnrtInvokeKernel" in source
    assert f"launch_{compilation.artifact.kernel_name}" in source


@pytest.mark.skipif(
    not _BANG_COMPILER_AVAILABLE, reason="cncc is not available"
)
@pytest.mark.skipif(not _HAS_MLU, reason="no MLU device is available")
def test_bang_aot_end_to_end(tmp_path):
    tensors = tuple(Tensor(shape=(257,), dtype=ninetoothed.float32) for _ in range(3))
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=tensors,
            backend="bang",
            kernel_name="bang_add",
            max_num_configs=1,
        )
    )
    handle = DEFAULT_COMPILER.materialize(
        compilation,
        output_dir=tmp_path,
        mode="aot",
    )

    input = torch.randn(257, device="mlu")
    other = torch.randn_like(input)
    output = torch.empty_like(input)
    handle(input, other, output)

    reloaded = load_built_artifact(handle._built_artifact)
    reloaded_output = torch.empty_like(input)
    reloaded(input, other, reloaded_output)

    assert handle._built_artifact.binary_path is not None
    assert torch.allclose(output, input + other)
    assert torch.allclose(reloaded_output, input + other)


@pytest.mark.skipif(
    not _BANG_COMPILER_AVAILABLE, reason="cncc is not available"
)
@pytest.mark.skipif(not _HAS_MLU, reason="no MLU device is available")
@pytest.mark.parametrize("shape", (1, 255, 256, 257, 4096))
def test_bang_jit_launch(shape):
    tensors = tuple(Tensor(shape=(shape,), dtype=ninetoothed.float32) for _ in range(3))
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=tensors,
            backend="bang",
            kernel_name=f"bang_jit_{shape}",
            max_num_configs=1,
        )
    )
    handle = DEFAULT_COMPILER.materialize(compilation, mode="jit")

    input = torch.randn(shape, device="mlu")
    other = torch.randn_like(input)
    output = torch.empty_like(input)
    result = handle(input, other, output)

    assert torch.allclose(output, input + other)
    if result is not None:
        assert torch.allclose(result, input + other)
