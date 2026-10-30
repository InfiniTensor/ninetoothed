import pytest

import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed.compiler import (
    DEFAULT_COMPILER,
    CompileRequest,
    load_built_artifact,
    lower,
)


def _arrangement(x, out):
    return x.tile((128,)), out.tile((128,))


def _increment(x, out):
    out = x + 1  # noqa: F841


def _scalar_arrangement(x, out, scale):
    return x.tile((128,)), out.tile((128,)), scale


def _scale(x, out, scale):
    out = x * scale  # noqa: F841


def _row_arrangement(x, out):
    return x.tile((1, -1)), out.tile((1,))


def _entropy(x, out):
    numerator = ntl.exp(x - ntl.max(x, axis=1)[:, None])
    probability = numerator / ntl.sum(numerator, axis=1)[:, None]
    out = -ntl.sum(probability * ntl.log(probability + 1e-8), axis=1)  # noqa: F841


def _atomic(x, out):
    ntl.atomic_add(out.data_ptr(), ntl.sum(x))


def _matrix_arrangement(x, y, out):
    return tuple(tensor.tile((8, 8)) for tensor in (x, y, out))


def _matmul(x, y, out):
    out = ntl.dot(x, y)  # noqa: F841


def test_ascendc_source_and_launch_contract():
    artifact = lower(
        _arrangement,
        _increment,
        (
            nt.Tensor(shape=(257,), dtype=nt.float32),
            nt.Tensor(shape=(257,), dtype=nt.float32),
        ),
        backend="ascendc",
        kernel_name="ascendc_baseline",
    )
    source = artifact.primary_source
    assert "__aicore__" in source
    assert "GetSubBlockIdx()!=0" in source
    assert "launch_ascendc_baseline" in source
    assert "<<<1, nullptr," in source

    with pytest.raises(ValueError, match="atomic updates"):
        lower(_arrangement, _atomic, (nt.Tensor(1), nt.Tensor(1)), backend="ascendc")

    artifact = lower(
        _scalar_arrangement,
        _scale,
        (
            nt.Tensor(shape=(257,), dtype=nt.float16),
            nt.Tensor(shape=(257,), dtype=nt.float16),
            nt.Tensor(0, dtype=nt.float16),
        ),
        backend="ascendc",
    )
    assert artifact.metadata["ascendc_host_scalar_abi"]["version"] == "uint16-half-v1"
    assert "::memcpy" in artifact.primary_source


def test_ascendc_jit_aot_reload_and_numerics(tmp_path):
    import torch

    pytest.importorskip("torch_npu")

    if not torch.npu.is_available():
        pytest.skip("NPU device required")

    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int64):
        compilation = DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=_arrangement,
                application=_increment,
                tensors=tuple(
                    nt.Tensor(shape=(257,), dtype=str(dtype)) for _ in range(2)
                ),
                backend="ascendc",
                max_num_configs=1,
            )
        )
        handle = DEFAULT_COMPILER.materialize(
            compilation, output_dir=tmp_path / str(dtype), mode="aot"
        )
        host = torch.arange(257, dtype=dtype)

        if dtype == torch.int64:
            host += 1 << 55

        x = host.to("npu")
        storage = torch.full((259,), -123, dtype=dtype).to("npu")
        out = storage[1:-1]

        for executable in (handle, load_built_artifact(handle._built_artifact)):
            executable(x, out)
            torch.npu.synchronize()
            torch.testing.assert_close(out.cpu(), host + 1, rtol=0, atol=0)
            assert storage[0].item() == storage[-1].item() == -123

    kernel = nt.make(
        _arrangement,
        _increment,
        (nt.Tensor(1, dtype=nt.float32), nt.Tensor(1, dtype=nt.float32)),
        backend="ascendc",
        max_num_configs=1,
    )
    x = torch.zeros(257).to("npu")

    for _ in range(8):
        kernel(x, x)

    torch.npu.synchronize()
    torch.testing.assert_close(x.cpu(), torch.full((257,), 8.0), rtol=0, atol=0)

    for dtype in (torch.float16, torch.bfloat16):
        tensors = (
            nt.Tensor(shape=(257,), dtype=str(dtype)),
            nt.Tensor(shape=(257,), dtype=str(dtype)),
            nt.Tensor(0, dtype=str(dtype)),
        )
        handle = DEFAULT_COMPILER.materialize(
            DEFAULT_COMPILER.compile(
                CompileRequest(
                    arrangement=_scalar_arrangement,
                    application=_scale,
                    tensors=tensors,
                    backend="ascendc",
                    max_num_configs=1,
                )
            ),
            output_dir=tmp_path / (str(dtype) + "-scalar"),
            mode="aot",
        )
        x = torch.ones(257, dtype=dtype).to("npu")
        out = torch.empty_like(x)
        load_built_artifact(handle._built_artifact)(x, out, 1.5)
        torch.npu.synchronize()
        torch.testing.assert_close(
            out.cpu(), torch.full((257,), 1.5, dtype=dtype), rtol=0, atol=0
        )

    host = torch.arange(7 * 30, dtype=torch.float32).reshape(7, 30) / 30
    x = host.to("npu")[:, 1:30:2]
    expected = host[:, 1:30:2].softmax(1)
    expected = -(expected * (expected + 1e-8).log()).sum(1)
    out = torch.empty(7, device="npu")
    kernel = nt.make(
        _row_arrangement,
        _entropy,
        (nt.Tensor(2), nt.Tensor(1)),
        backend="ascendc",
        max_num_configs=1,
    )
    kernel(x, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected)
    left = torch.arange(35, dtype=torch.float32).reshape(5, 7) / 16
    right = torch.arange(21, dtype=torch.float32).reshape(7, 3) / 32
    output = torch.empty((5, 3), device="npu")
    kernel = nt.make(
        _matrix_arrangement,
        _matmul,
        (nt.Tensor(2), nt.Tensor(2), nt.Tensor(2)),
        backend="ascendc",
        max_num_configs=1,
    )
    kernel(left.to("npu"), right.to("npu"), output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), left @ right)
