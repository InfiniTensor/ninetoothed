import pytest

import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest, load_built_artifact


def _arrangement(x, y, out):
    return tuple(tensor.tile((128,)) for tensor in (x, y, out))


def _application(x, y, out):
    out = (x + y) * y  # noqa: F841


def _row_arrangement(x, out):
    return x.tile((1, -1)), out.tile((1,))


def _maximum(x, out):
    out = ntl.max(x, axis=1)  # noqa: F841


def _minimum(x, out):
    out = ntl.min(x, axis=1)  # noqa: F841


def _entropy(x, out):
    maximum = ntl.max(x, axis=1)
    numerator = ntl.exp(x - maximum[:, None])
    probability = numerator / ntl.sum(numerator, axis=1)[:, None]
    out = -ntl.sum(probability * ntl.log(probability + 1e-8), axis=1)  # noqa: F841


def _atomic_arrangement(x, out):
    return x.tile((128,)), out.tile((1,))


def _atomic_sum(x, out):
    ntl.atomic_add(out.data_ptr(), ntl.sum(x))


def _normalize_arrangement(x, out):
    return x.tile((1, -1)), out.tile((1, -1))


def _normalize(x, out):
    mean = ntl.sum(x, axis=1, keepdims=True) / x.shape[1]
    centered = x - mean
    variance = ntl.sum(centered * centered, axis=1, keepdims=True) / x.shape[1]
    out = centered * ntl.rsqrt(variance + 1e-5)  # noqa: F841


def _loop_arrangement(x, y, z, out):
    return tuple(tensor.tile((8, 8)) for tensor in (x, y, z, out))


def _loop_matmul(x, y, z, out):
    value = ntl.zeros(out.shape, dtype=ntl.float32)

    for _ in range(2):
        value += ntl.dot(ntl.dot(x, y), z)

    out = value  # noqa: F841


def test_bangc_scalar_source_contract():
    from ninetoothed.compiler import lower

    artifact = lower(
        _arrangement,
        _application,
        tuple(nt.Tensor(1) for _ in range(3)),
        backend="bangc",
        kernel_name="bangc_baseline",
    )
    source = artifact.primary_source
    assert "__mlu_entry__" in source
    assert "cnrtGetLastError" in source
    assert "launch_bangc_baseline" in source
    assert "__bang_add(" not in source
    assert artifact.backend.value == "bangc"
    artifact = lower(
        _loop_arrangement,
        _loop_matmul,
        tuple(nt.Tensor(shape=(8, 8), dtype=nt.float32) for _ in range(4)),
        backend="bangc",
        kernel_name="bangc_bounded_loop",
    )
    assert "__nram__" in artifact.primary_source
    artifact = lower(
        _row_arrangement,
        _entropy,
        (
            nt.Tensor(shape=(8, 32), dtype=nt.float32),
            nt.Tensor(shape=(8,), dtype=nt.float32),
        ),
        backend="bangc",
        kernel_name="bangc_bounded_reductions",
    )
    assert "__nram__" in artifact.primary_source


def test_bangc_jit_aot_reload_and_numerics(tmp_path):
    import torch

    pytest.importorskip("torch_mlu")

    if not torch.mlu.is_available():
        pytest.skip("MLU device required")

    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int64):
        tensors = tuple(nt.Tensor(shape=(257,), dtype=str(dtype)) for _ in range(3))
        compilation = DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=_arrangement,
                application=_application,
                tensors=tensors,
                backend="bangc",
                max_num_configs=1,
            )
        )
        handle = DEFAULT_COMPILER.materialize(
            compilation, output_dir=tmp_path / str(dtype), mode="aot"
        )
        host = torch.arange(257, dtype=dtype)

        if dtype == torch.int64:
            host += 1 << 54

        x, y = host.to("mlu"), torch.ones(257, dtype=dtype).to("mlu")
        storage = torch.full((259,), -123, dtype=dtype).to("mlu")
        out = storage[1:-1]
        handle(x, y, out)
        torch.mlu.synchronize()
        torch.testing.assert_close(out.cpu(), host + 1, rtol=0, atol=0)
        assert storage[0].item() == storage[-1].item() == -123
        reloaded = load_built_artifact(handle._built_artifact)
        out.zero_()
        reloaded(x, y, out)
        torch.mlu.synchronize()
        torch.testing.assert_close(out.cpu(), host + 1, rtol=0, atol=0)

    host = torch.tensor([[2**55 + 1, 2**55 + 3], [-(2**55) - 3, -(2**55) - 1]])
    x = host.to("mlu")
    out = torch.empty(2, dtype=torch.int64).to("mlu")

    for application, expected in (
        (_maximum, host.amax(1)),
        (_minimum, host.amin(1)),
    ):
        kernel = nt.make(
            _row_arrangement,
            application,
            (nt.Tensor(2), nt.Tensor(1)),
            backend="bangc",
            max_num_configs=1,
        )
        kernel(x, out)
        torch.mlu.synchronize()
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    kernel = nt.make(
        _atomic_arrangement,
        _atomic_sum,
        (nt.Tensor(1), nt.Tensor(1)),
        backend="bangc",
        max_num_configs=1,
    )
    out = torch.tensor([2**55 + 1], dtype=torch.int64).to("mlu")
    kernel(torch.ones(128, dtype=torch.int64).to("mlu"), out)
    torch.mlu.synchronize()
    assert out.cpu().item() == 2**55 + 129

    host = torch.arange(256, dtype=torch.float32).reshape(8, 32) / 32
    x, out = host.to("mlu"), torch.empty_like(host).to("mlu")
    kernel = nt.make(
        _normalize_arrangement,
        _normalize,
        (nt.Tensor(2), nt.Tensor(2)),
        backend="bangc",
        max_num_configs=1,
    )
    kernel(x, out)
    expected = (host - host.mean(1, keepdim=True)) / torch.sqrt(
        host.var(1, unbiased=False, keepdim=True) + 1e-5
    )
    torch.mlu.synchronize()
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-5)
    kernel = nt.make(
        _row_arrangement,
        _entropy,
        (
            nt.Tensor(shape=(8, 32), dtype=nt.float32),
            nt.Tensor(shape=(8,), dtype=nt.float32),
        ),
        backend="bangc",
        max_num_configs=1,
    )
    entropy = torch.empty(8, dtype=torch.float32).to("mlu")
    kernel(x, entropy)
    torch.mlu.synchronize()
    probability = host.softmax(1)
    expected = -(probability * (probability + 1e-8).log()).sum(1)
    torch.testing.assert_close(entropy.cpu(), expected)
    host = host[:7, 1:30:2]
    x = host.to("mlu")
    kernel = nt.make(
        _row_arrangement,
        _entropy,
        (nt.Tensor(2), nt.Tensor(1)),
        backend="bangc",
        max_num_configs=1,
    )
    probability = host.softmax(1)
    expected = -(probability * (probability + 1e-8).log()).sum(1)
    # An odd row width exercises masked tile elements; preserve input strides.
    strided = torch.empty((7, 30), dtype=host.dtype).to("mlu")[:, 1:30:2]
    strided.copy_(x)
    entropy = torch.empty(7, dtype=host.dtype).to("mlu")
    kernel(strided, entropy)
    torch.mlu.synchronize()
    torch.testing.assert_close(entropy.cpu(), expected)
    # A row-local output alias must take the conservative scalar fallback.
    kernel(strided, strided[:, 0])
    torch.mlu.synchronize()
    torch.testing.assert_close(strided[:, 0].cpu(), expected)
    host = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
    x = host.to("mlu")
    out = torch.empty_like(x)
    kernel = nt.make(
        _loop_arrangement,
        _loop_matmul,
        tuple(nt.Tensor(shape=(8, 8), dtype=nt.float32) for _ in range(4)),
        backend="bangc",
        max_num_configs=1,
    )
    kernel(x, x, x, out)
    torch.mlu.synchronize()
    torch.testing.assert_close(out.cpu(), 2 * (host @ host @ host))
