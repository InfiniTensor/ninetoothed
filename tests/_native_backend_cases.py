"""Core DSL workloads and checks shared by native C backends."""

import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest, load_built_artifact


def _arrangement(*tensors):
    return tuple(tensor.tile((128,)) for tensor in tensors)


def _application(x, y, out):
    out = (x + y) * y  # noqa: F841


def _row_arrangement(x, out):
    return x.tile((1, -1)), out.tile((1,))


def _maximum(x, out):
    out = ntl.max(x, axis=1)  # noqa: F841


def _minimum(x, out):
    out = ntl.min(x, axis=1)  # noqa: F841


def _summation(x, out):
    out = ntl.sum(x, axis=1)  # noqa: F841


def _entropy(x, out):
    maximum = ntl.max(x, axis=1)
    numerator = ntl.exp(x - maximum[:, None])
    probability = numerator / ntl.sum(numerator, axis=1)[:, None]
    out = -ntl.sum(probability * ntl.log(probability + 1e-8), axis=1)  # noqa: F841


def _indexed_arrangement(*tensors):
    return tuple(tensor.tile((8,)) for tensor in tensors)


def _conditional(x, out):
    total = ntl.sum(x)
    maximum = ntl.max(x)
    value = x

    if x > 0:
        value = x + total
    else:
        value = x + maximum

    out = value  # noqa: F841


def _indexed(indices, x, out):
    index = indices + 0
    used = index + 1
    later = indices + 2
    out[index] = x + ntl.sum(x) + ntl.sum(x) + 0 * used + 0 * later


def _cast_absolute(x, out):
    out = ntl.abs(x).to(out.dtype)  # noqa: F841


def _loop_arrangement(x, y, z, out):
    return tuple(tensor.tile((8, 8)) for tensor in (x, y, z, out))


def _loop_matmul(x, y, z, out):
    value = ntl.zeros(out.shape, dtype=ntl.float32)

    for _ in range(2):
        value += ntl.dot(ntl.dot(x, y), z)

    out = value  # noqa: F841


def _mixed_compare(x, y, out):
    out = ntl.where(x > y, 1, 0)  # noqa: F841


def check_native_baseline(tmp_path, *, backend, device, extra_integer_pairs=()):
    import torch

    runtime = getattr(torch, device)

    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int64):
        tensors = tuple(nt.Tensor(shape=(257,), dtype=str(dtype)) for _ in range(3))
        compilation = DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=_arrangement,
                application=_application,
                tensors=tensors,
                backend=backend,
                max_num_configs=1,
            )
        )
        handle = DEFAULT_COMPILER.materialize(
            compilation, output_dir=tmp_path / str(dtype), mode="aot"
        )
        host = torch.arange(257, dtype=dtype)

        if dtype == torch.int64:
            host += 1 << 54
            other = torch.arange(257, dtype=dtype) % 3 + 1
        else:
            host = torch.linspace(-2, 2, 257, dtype=dtype)
            other = torch.linspace(0.5, 1.5, 257, dtype=dtype)

        expected = (host + other) * other
        x, y = host.to(device), other.to(device)

        for executable in (handle, load_built_artifact(handle._built_artifact)):
            storage = torch.full((259,), -123, dtype=dtype).to(device)
            out = storage[1:-1]
            executable(x, y, out)
            runtime.synchronize()
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
            assert storage[0].item() == storage[-1].item() == -123

    values = torch.tensor([[2**55 + 1, 2**55 + 3], [-(2**55) - 3, -(2**55) - 1]])

    for dtype in (torch.int64, torch.bfloat16):
        host = values.to(dtype)
        x = host.to(device)
        out = torch.empty(2, dtype=dtype).to(device)

        for application, expected in (
            (_maximum, host.amax(1)),
            (_minimum, host.amin(1)),
        ):
            kernel = nt.make(
                _row_arrangement,
                application,
                (nt.Tensor(2), nt.Tensor(1)),
                backend=backend,
                max_num_configs=1,
            )
            kernel(x, out)
            runtime.synchronize()
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    host = torch.ones(2, 258, dtype=torch.bfloat16)
    out = torch.empty(2, dtype=torch.bfloat16).to(device)
    kernel = nt.make(
        _row_arrangement,
        _summation,
        (nt.Tensor(2), nt.Tensor(1)),
        backend=backend,
        max_num_configs=1,
    )
    kernel(host.to(device), out)
    runtime.synchronize()
    torch.testing.assert_close(out.cpu(), host.sum(1), rtol=0, atol=0)
    base = torch.arange(256, dtype=torch.float32).reshape(8, 32)[:7, :30] / 32
    host = base[:, 1:30:2]
    kernel = nt.make(
        _row_arrangement,
        _entropy,
        (nt.Tensor(2), nt.Tensor(1)),
        backend=backend,
        max_num_configs=1,
    )
    probability = host.softmax(1)
    expected = -(probability * (probability + 1e-8).log()).sum(1)
    # An odd row width exercises masked tile elements; preserve input strides.
    device_base = base.to(device)
    strided = device_base[:, 1:30:2]
    entropy = torch.empty(7, dtype=host.dtype).to(device)
    kernel(strided, entropy)
    runtime.synchronize()
    torch.testing.assert_close(entropy.cpu(), expected)
    # A row-local output alias must take the conservative scalar fallback.
    kernel(strided, strided[:, 0])
    runtime.synchronize()
    expected_base = base.clone()
    expected_base[:, 1] = expected
    torch.testing.assert_close(device_base.cpu(), expected_base)
    host = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
    x = host.to(device)
    out = torch.empty_like(x)
    kernel = nt.make(
        _loop_arrangement,
        _loop_matmul,
        tuple(nt.Tensor(shape=(8, 8), dtype=nt.float32) for _ in range(4)),
        backend=backend,
        max_num_configs=1,
    )
    kernel(x, x, x, out)
    runtime.synchronize()
    torch.testing.assert_close(out.cpu(), 2 * (host @ host @ host))

    host = torch.tensor([-4, -3, -2, -1, 1, 2, 3, 4], dtype=torch.float32)
    x, out = host.to(device), torch.empty_like(host).to(device)
    kernel = nt.make(
        _arrangement,
        _conditional,
        (nt.Tensor(1), nt.Tensor(1)),
        backend=backend,
        max_num_configs=1,
    )
    kernel(x, out)
    runtime.synchronize()
    torch.testing.assert_close(
        out.cpu(), torch.where(host > 0, host + host.sum(), host + host.max())
    )
    kernel = nt.make(
        _indexed_arrangement,
        _indexed,
        (nt.Tensor(1), nt.Tensor(1), nt.Tensor(1)),
        backend=backend,
        max_num_configs=1,
    )
    kernel(torch.arange(8).flip(0).to(device), x, out)
    runtime.synchronize()
    torch.testing.assert_close(out.cpu(), (host + 2 * host.sum()).flip(0))

    for dtype in (torch.int8, torch.int16, torch.float32):
        values = (
            [42, -42]
            if dtype != torch.float32
            else [6442450687, -6442451201, 72057598332895233]
        )
        host = torch.tensor(values, dtype=torch.int64)
        out = torch.empty(len(values), dtype=dtype, device=device)
        kernel = nt.make(
            _arrangement,
            _cast_absolute,
            (nt.Tensor(1), nt.Tensor(1)),
            backend=backend,
            max_num_configs=1,
        )
        kernel(host.to(device), out)
        runtime.synchronize()
        torch.testing.assert_close(out.cpu(), host.abs().to(dtype), rtol=0, atol=0)

    pairs = (
        (
            torch.tensor([127, -128, 0], dtype=torch.int8),
            torch.tensor([128, -129, 0], dtype=torch.int16),
        ),
        (
            torch.tensor([255, 0, 1], dtype=torch.uint8),
            torch.tensor([-1, -1, 1], dtype=torch.int8),
        ),
    ) + extra_integer_pairs

    for x, y in pairs:
        for left, right in ((x, y), (y, x)):
            out = torch.empty(3).to(device)
            nt.make(
                _arrangement,
                _mixed_compare,
                (
                    nt.Tensor(1, dtype=str(left.dtype)),
                    nt.Tensor(1, dtype=str(right.dtype)),
                    nt.Tensor(1),
                ),
                backend=backend,
                max_num_configs=1,
            )(left.to(device), right.to(device), out)
            runtime.synchronize()
            expected = torch.tensor(
                [float(int(a) > int(b)) for a, b in zip(left.tolist(), right.tolist())]
            )
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
