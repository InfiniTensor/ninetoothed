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


def _compare(x, out):
    out = ntl.where(x > 0, x, ntl.where(x < 0, -x, ntl.where(x == 0, 1, 100)))  # noqa: F841


def _absolute(x, out):
    out = ntl.abs(x)  # noqa: F841


def _where_boundary(x, out):
    out = ntl.where(x > 0, x, 257) - 256  # noqa: F841


def _math(x, logarithm, exponential, tangent, exp, sinh, cosh, error):
    logarithm = ntl.log1p(x)  # noqa: F841
    exponential = ntl.expm1(x)  # noqa: F841
    tangent = ntl.tanh(x)  # noqa: F841
    exp = ntl.exp(x)  # noqa: F841
    sinh = ntl.sinh(x)  # noqa: F841
    cosh = ntl.cosh(x)  # noqa: F841
    error = ntl.erf(x)  # noqa: F841


def _live_tiles(x, out):
    y = ntl.abs(x)
    z = ntl.abs(x + 1)
    out = ntl.sum(y) + ntl.max(z)  # noqa: F841


def _tile_arrangement(x, out):
    return x.tile((8,)), out.tile((1,))


def _tensor_if(x, out):
    total = ntl.sum(x)
    maximum = ntl.max(x)
    value = x

    if x > 0:
        value = x + total
    else:
        value = x + maximum

    out = value  # noqa: F841


def _power(x, y, out):
    out = ntl.pow(x, y)  # noqa: F841


def _mixed_compare(x, y, out):
    out = ntl.where(x > y, 1, 0)  # noqa: F841


def _power_arrangement(x, y, out):
    return tuple(tensor.tile((32,)) for tensor in (x, y, out))


def _math_arrangement(x, *outputs):
    return tuple(tensor.tile((32,)) for tensor in (x, *outputs))


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
    assert "DataCopyPad" in source

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
    assert "nt_trig_remainder" not in source


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

    streams = (torch.npu.Stream(), torch.npu.Stream())
    inputs = tuple(torch.full((8,), float(i)).to("npu") for i in range(4))

    for iteration in range(50):
        storage = torch.full((18,), -123.0).to("npu")
        torch.npu.synchronize()

        for i, stream in enumerate(streams):
            with torch.npu.stream(stream):
                kernel(inputs[2 * i + iteration % 2], storage[1 + 8 * i : 9 + 8 * i])

        torch.npu.synchronize()
        expected = torch.cat(
            tuple(torch.full((8,), float(2 * i + iteration % 2 + 1)) for i in range(2))
        )
        torch.testing.assert_close(storage[1:-1].cpu(), expected, rtol=0, atol=0)
        assert storage[0].item() == storage[-1].item() == -123

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

    host = torch.tensor([-0.75, -0.5, -0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.bfloat16)
    out = torch.empty_like(host).to("npu")
    nt.make(
        _arrangement,
        _compare,
        (nt.Tensor(1, dtype=nt.bfloat16),) * 2,
        backend="ascendc",
        max_num_configs=1,
    )(host.to("npu"), out)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu(), torch.where(host == 0, 1, host.abs()), rtol=0, atol=0
    )
    nt.make(
        _arrangement,
        _where_boundary,
        (nt.Tensor(1, dtype=nt.bfloat16),) * 2,
        backend="ascendc",
        max_num_configs=1,
    )(host.to("npu"), out)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu(), torch.where(host > 0, host, 257) - 256, rtol=0, atol=0
    )

    for dtype, large in ((torch.int32, 2**24 + 1), (torch.int64, 2**55 + 1)):
        host = torch.tensor([large, -large, torch.iinfo(dtype).min], dtype=dtype)
        out = torch.empty_like(host).to("npu")
        nt.make(
            _arrangement,
            _absolute,
            (nt.Tensor(1, dtype=str(dtype)),) * 2,
            backend="ascendc",
            max_num_configs=1,
        )(host.to("npu"), out)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), host.abs(), rtol=0, atol=0)

    host = torch.tensor(
        [
            -float("inf"),
            -89,
            -1,
            -0.5,
            -1e-7,
            -1e-8,
            -0.0,
            0.0,
            1e-8,
            1e-7,
            0.5,
            1,
            88.721,
            89,
            float("inf"),
            float("nan"),
        ]
    )
    outputs = tuple(torch.empty_like(host).to("npu") for _ in range(7))
    nt.make(
        _math_arrangement,
        _math,
        (nt.Tensor(1, dtype=nt.float32),) * 8,
        backend="ascendc",
        max_num_configs=1,
    )(host.to("npu"), *outputs)
    torch.npu.synchronize()

    for actual, function in zip(
        outputs,
        (
            torch.log1p,
            torch.expm1,
            torch.tanh,
            torch.exp,
            torch.sinh,
            torch.cosh,
            torch.erf,
        ),
    ):
        expected = function(host.double()).float()
        torch.testing.assert_close(
            actual.cpu(), expected, rtol=2e-6, atol=0, equal_nan=True
        )
        zeros = expected == 0
        assert torch.equal(
            torch.signbit(actual.cpu()[zeros]), torch.signbit(expected[zeros])
        )

    for dtype, declaration in ((torch.int64, nt.int64), (torch.uint64, nt.uint64)):
        host = torch.arange(1, 9).to(dtype)
        output = torch.empty(1, dtype=dtype).to("npu")
        nt.make(
            _tile_arrangement,
            _live_tiles,
            (
                nt.Tensor(shape=(8,), dtype=declaration),
                nt.Tensor(shape=(1,), dtype=declaration),
            ),
            backend="ascendc",
            max_num_configs=1,
        )(host.to("npu"), output)
        torch.npu.synchronize()
        assert output.cpu().item() == 45

    host = torch.arange(-4, 4, dtype=torch.float32)
    output = torch.empty_like(host).to("npu")
    nt.make(
        _arrangement,
        _tensor_if,
        (nt.Tensor(1),) * 2,
        backend="ascendc",
        max_num_configs=1,
    )(host.to("npu"), output)
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu(), torch.where(host > 0, host + host.sum(), host + host.max())
    )

    left = torch.tensor(
        [
            torch.finfo(torch.float32).max,
            1.8446742974197924e19,
            4294967040.0,
            -3.0,
            1625.4986572265625,
            5.232202238403261e-05,
        ]
    )
    right = torch.tensor([1.0, 2.0, 4.0, 3.0, 12.0, -9.0])
    output = torch.empty_like(left).to("npu")
    nt.make(
        _power_arrangement,
        _power,
        (nt.Tensor(1),) * 3,
        backend="ascendc",
        max_num_configs=1,
    )(left.to("npu"), right.to("npu"), output)
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu(), left.double().pow(right.double()).float(), rtol=2e-6, atol=0
    )

    pairs = (
        (
            torch.tensor([127, -128, 0], dtype=torch.int8),
            torch.tensor([128, -129, 0], dtype=torch.int16),
        ),
        (
            torch.tensor([255, 0, 1], dtype=torch.uint8),
            torch.tensor([256, 0, 2], dtype=torch.uint16),
        ),
        (
            torch.tensor([255, 0, 1], dtype=torch.uint8),
            torch.tensor([-1, -1, 1], dtype=torch.int8),
        ),
    )

    for x, y in pairs:
        for left, right in ((x, y), (y, x)):
            out = torch.empty(3).to("npu")
            nt.make(
                _power_arrangement,
                _mixed_compare,
                (
                    nt.Tensor(1, dtype=str(left.dtype)),
                    nt.Tensor(1, dtype=str(right.dtype)),
                    nt.Tensor(1),
                ),
                backend="ascendc",
                max_num_configs=1,
            )(left.to("npu"), right.to("npu"), out)
            torch.npu.synchronize()
            expected = torch.tensor(
                [float(int(a) > int(b)) for a, b in zip(left.tolist(), right.tolist())]
            )
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
