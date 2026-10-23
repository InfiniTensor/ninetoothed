import pytest
import torch

import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed.compiler.runtime import _runtime_wrapper
from ninetoothed.ir import LaunchABI, LaunchBinding
from tests.test_attention import arrangement as attention_arrangement


def _require_tilelang():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TileLang execution")
    return pytest.importorskip("tilelang")


def _row_arrangement(x, out):
    return x.tile((2, 512)), out.tile((2,))


def _row_sum(x, out):
    out = ntl.sum(ntl.exp(x), axis=1)  # noqa: F841


def _vector_arrangement(x, out):
    return x.tile((512,)), out.tile((512,))


def _scalar_sum(x, out):
    out = ntl.sum(x)  # noqa: F841


def test_tilelang_reduction_domain_and_storage():
    _require_tilelang()
    x = torch.zeros((2, 300), device="cuda")
    expected = x.exp().sum(1)
    out = torch.full_like(expected, -999)
    kernel = nt.make(
        _row_arrangement,
        _row_sum,
        (
            nt.Tensor(shape=x.shape, dtype="float32"),
            nt.Tensor(shape=out.shape, dtype="float32"),
        ),
        backend="tilelang",
        max_num_configs=1,
    )
    kernel(x, out)
    torch.testing.assert_close(out, expected)
    assert "T.reduce_sum(" in kernel._artifact.primary_source

    # A scalar reduction must fill the entire destination tile on both backends.
    x = torch.arange(512, device="cuda", dtype=torch.float32)

    for backend in ("cuda", "tilelang"):
        out = torch.full_like(x, float("nan"))
        kernel = nt.make(
            _vector_arrangement,
            _scalar_sum,
            tuple(nt.Tensor(shape=x.shape, dtype="float32") for _ in range(2)),
            backend=backend,
            max_num_configs=1,
        )
        kernel(x, out)
        torch.testing.assert_close(out, x.sum().expand_as(x))


def _attention_arrangement(q, k, v, is_causal, offset, out):
    arranged = attention_arrangement(
        q, k, v, is_causal, out, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64
    )

    return (*arranged[:-1], offset, arranged[-1])


def _attention(q, k, v, is_causal, offset, out):
    q_loaded = (q * 1.44269504089).to(q.dtype)
    acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
    denominator = ntl.full((q.shape[-2],), 1, dtype=ntl.float32)
    maximum = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)

    for i in range(k.shape[0]):
        scores = ntl.dot(q_loaded, ntl.trans(k[i]))
        scores = ntl.where(k[i].offsets(-2) < k.source.shape[-2], scores, float("-inf"))

        if is_causal:
            mask = q.offsets(-2)[:, None] >= k[i].offsets(-2)[None, :]
            scores = ntl.where(mask, scores, float("-inf"))

        current = ntl.maximum(maximum, ntl.max(scores, 1))
        probabilities = ntl.exp2(scores - current[:, None])
        correction = ntl.exp2(maximum - current)
        acc = acc * correction[:, None] + ntl.dot(probabilities.to(v[i].dtype), v[i])
        denominator = denominator * correction + ntl.sum(probabilities, 1)
        maximum = current

    out = acc / denominator[:, None] + offset  # noqa: F841


def test_tilelang_attention_preserves_application():
    _require_tilelang()
    length, causal, offset = 17, True, 7
    shape = (1, 1, length, 16)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3))
    out = torch.empty_like(q)
    tensors = tuple(nt.Tensor(shape=shape, dtype="float16") for _ in range(4))
    kernel = nt.make(
        _attention_arrangement,
        _attention,
        (
            *tensors[:3],
            nt.Tensor(0, constexpr=True),
            nt.Tensor(0, constexpr=True),
            tensors[3],
        ),
        backend="tilelang",
        max_num_configs=1,
    )
    kernel(q, k, v, causal, offset, out)
    assert "Schedule: materialized SSA tiles" in kernel._artifact.primary_source
    plan = kernel._artifact.metadata["tilelang_emission"]
    assert plan["actual"] == "materialized-tiles"
    assert kernel._artifact.metadata["launch_grid"] == (plan["grid"],)
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=1
        )
        + offset
    )
    torch.testing.assert_close(out, expected, rtol=0.02, atol=0.02)


def _dot_tiles(lhs, rhs, out):
    return lhs.tile((64, 64)), rhs.tile((64, 64)), out.tile((64, 64))


def _dot_reduce(lhs, rhs, out):
    product = ntl.dot(lhs, rhs)
    out = product + ntl.sum(product, axis=1)[:, None]  # noqa: F841


def test_tilelang_dot_reduction_uses_materialized_tiles():
    _require_tilelang()
    # Offset views also exercise the scalar transfer path without cloning.
    a, b = (
        torch.randn(64 * 64 + 1, device="cuda", dtype=torch.float16)[1:].view(64, 64)
        for _ in range(2)
    )
    a.mul_(0.1)
    b.mul_(0.1)
    out = torch.full_like(a, float("nan"))
    product = a.float() @ b.float()
    expected = product + product.sum(1)[:, None]
    kernel = nt.make(
        _dot_tiles,
        _dot_reduce,
        (nt.Tensor(2), nt.Tensor(2), nt.Tensor(2)),
        backend="tilelang",
        max_num_configs=1,
    )
    kernel(a, b, out)
    torch.testing.assert_close(out, expected.to(out.dtype), rtol=0.01, atol=0.01)
    plan = kernel._artifact.metadata["tilelang_emission"]
    assert plan["actual"] == "materialized-tiles"


def test_runtime_flattening_preserves_aliases():
    abi = LaunchABI(
        public_args=("a", "b"),
        outputs=("a",),
        kernel_args=tuple(
            LaunchBinding(name=name, source=name, kind="tensor", access="read_write")
            for name in ("a", "b")
        ),
    )

    def update(a, b):
        a.add_(1)
        b.add_(1)

    storage = torch.zeros(3)
    view = storage[1:2]
    launch = _runtime_wrapper(update, abi)
    launch(view, view)
    torch.testing.assert_close(storage, torch.tensor([0.0, 2.0, 0.0]))
    prepared = launch._ninetoothed_prepare((view, view), {})
    prepared.invoker(prepared, (view, view), {})
    torch.testing.assert_close(storage, torch.tensor([0.0, 4.0, 0.0]))


def test_tilelang_handles_rebind_current_stream():
    _require_tilelang()
    import tvm

    from ninetoothed.backends.materializers.tilelang import _tilelang_streamed_call

    get_stream = tvm.get_global_func("runtime.get_cuda_stream", allow_missing=True)

    if get_stream is None:
        pytest.skip("TVM does not expose its current CUDA stream")

    def current():
        value = get_stream()

        return int(getattr(value, "value", value) or 0)

    device = tvm.cuda(torch.cuda.current_device())
    original = current()
    observed = []
    first = _tilelang_streamed_call(lambda: observed.append(current()))
    second = _tilelang_streamed_call(lambda: observed.append(current()))
    a, b = torch.cuda.Stream(), torch.cuda.Stream()

    try:
        with torch.cuda.stream(a):
            first()

        with torch.cuda.stream(b):
            second()

        with torch.cuda.stream(a):
            first()

        assert observed == [a.cuda_stream, b.cuda_stream, a.cuda_stream]
    finally:
        device.set_raw_stream(original)
