import functools
import pickle
import subprocess
import sys
import uuid

import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
import tests.test_matmul as matmul
from ninetoothed import Tensor
from ninetoothed.compiler import (
    DEFAULT_COMPILER,
    CompileRequest,
    load_built_artifact,
)
from tests.utils import get_available_devices


def test_tilelang_matmul_on_unaligned_views(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TileLang execution")

    arrangement = functools.partial(
        matmul.arrangement, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32
    )

    def application(lhs, rhs, output):
        accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

        for k in range(lhs.shape[0]):
            accumulator += ntl.dot(lhs[k], rhs[k])

        output = accumulator  # noqa: F841

    kernel = ninetoothed.make(
        arrangement,
        application,
        tuple(Tensor(2) for _ in range(3)),
        backend="tilelang",
        max_num_configs=1,
        output_dir=tmp_path,
        kernel_name="tilelang_unaligned_matmul",
    )

    for size, input_offset, output_offset in (
        (32, 0, 0),
        (32, 1, 0),
        (32, 0, 1),
        (35, 1, 1),
    ):
        count = size * size
        lhs = torch.randn(count + 1, device="cuda", dtype=torch.float16)
        rhs = torch.randn_like(lhs)
        output = torch.full_like(lhs, -123, dtype=torch.bfloat16)
        lhs_view = lhs[input_offset : input_offset + count].reshape(size, size)
        rhs_view = rhs[input_offset : input_offset + count].reshape(size, size)
        output_view = output[output_offset : output_offset + count].reshape(size, size)
        expected = output.clone()
        expected[output_offset : output_offset + count] = (
            lhs_view.float() @ rhs_view.float()
        ).ravel()
        kernel(lhs_view, rhs_view, output_view)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
        assert torch.equal(output[:output_offset], expected[:output_offset])
        assert torch.equal(
            output[output_offset + count :], expected[output_offset + count :]
        )

    reloaded = load_built_artifact(kernel._built_artifact)
    output.fill_(-123)
    reloaded(lhs_view, rhs_view, output_view)
    torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)


def _arrangement(input, other, output):
    return tuple(tensor.tile((64,)) for tensor in (input, other, output))


def _application(input, other, output):
    output = input + other  # noqa: F841


@pytest.mark.parametrize("device", get_available_devices(backend="triton"))
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
    input = torch.randn(258, device=device)[1:]
    other = torch.randn(258, device=device)[1:]
    output_storage = torch.full((259,), -123.0, device=device)
    output = output_storage[1:-1]
    handle(input, other, output)
    torch.testing.assert_close(output, input + other)
    assert output_storage[0].item() == output_storage[-1].item() == -123.0

    reloaded = load_built_artifact(handle._built_artifact)
    reloaded_storage = torch.full((259,), -123.0, device=device)
    reloaded_output = reloaded_storage[1:-1]
    reloaded(input, other, reloaded_output)
    assert reloaded_storage[0].item() == reloaded_storage[-1].item() == -123.0

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


def test_triton_aot_handle_is_reusable_across_cuda_contexts(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Triton multi-context testing requires at least 2 CUDA devices")

    capabilities = tuple(torch.cuda.get_device_capability(device) for device in (0, 1))

    if capabilities[0] != capabilities[1]:
        pytest.skip("Triton multi-context testing requires matching CUDA capabilities")

    tensors = tuple(Tensor(shape=(257,), dtype=ninetoothed.float32) for _ in range(3))
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=tensors,
            backend="triton",
            kernel_name=f"triton_multi_context_{uuid.uuid4().hex}",
            max_num_configs=1,
        )
    )
    handle = DEFAULT_COMPILER.materialize(
        compilation,
        output_dir=tmp_path,
        mode="aot",
    )

    def check_launch(launch, device):
        with torch.cuda.device(device):
            input = torch.randn(257, device=device)
            other = torch.randn_like(input)
            output = torch.empty_like(input)
            launch(input, other, output)
            torch.cuda.synchronize()
            torch.testing.assert_close(output, input + other)

    for device in (0, 1, 0):
        check_launch(handle, device)

    reloaded = load_built_artifact(handle._built_artifact)

    for device in (1, 0, 1):
        check_launch(reloaded, device)


@pytest.mark.parametrize("device", get_available_devices(backend="triton"))
@pytest.mark.parametrize("mode", ("jit", "aot"))
def test_cuda_empty_tensor_is_a_no_op(mode, device, tmp_path):
    tensors = tuple(Tensor(1, dtype=ninetoothed.float32) for _ in range(3))
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=tensors,
            backend="cuda",
            kernel_name=f"empty_cuda_{mode}",
        )
    )
    handle = DEFAULT_COMPILER.materialize(
        compilation,
        output_dir=tmp_path if mode == "aot" else None,
        mode=mode,
    )
    input = torch.empty(0, device=device)
    other = torch.empty_like(input)
    output = torch.empty_like(input)

    assert handle(input, other, output) is output
    torch.cuda.synchronize()

    if mode == "aot":
        reloaded = load_built_artifact(handle._built_artifact)
        reloaded_output = torch.empty_like(input)
        assert reloaded(input, other, reloaded_output) is reloaded_output
        torch.cuda.synchronize()


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


def test_tilelang_serial_reduction_on_strided_views():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TileLang execution")

    def arrangement(input, output):
        return input.tile((1, -1)), output.tile((1,))

    def application(input, output):
        output = ntl.sum(input, axis=1)  # noqa: F841

    kernel = ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(1)),
        backend="tilelang",
        max_num_configs=1,
    )
    input = (torch.arange(8 * 17, device="cuda", dtype=torch.int32) + 2**24).reshape(
        8, 17
    )[1:7:2, 1:11:2]
    storage = torch.full((9,), -123, device="cuda", dtype=torch.int32)
    output = storage[1:7:2]
    kernel(input, output)
    expected = torch.full_like(storage, -123)
    expected[1:7:2] = input.sum(1, dtype=torch.int32)
    torch.testing.assert_close(storage, expected, atol=0, rtol=0)
