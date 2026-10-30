import gc
import weakref

import pytest
import torch

import ninetoothed as nt
import ninetoothed.language as ntl

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _arrangement(*tensors):
    return tuple(tensor.tile((128,)) for tensor in tensors)


def _extrema(nt_max, nt_min, stream, minimum):
    stream = ntl.maximum(nt_max, nt_min)  # noqa: F841
    minimum = ntl.minimum(nt_max, nt_min)  # noqa: F841


def _rounded_arithmetic(x, y, z, out):
    out = (x + y) * z  # noqa: F841


def _positive_sum(x, out):
    out = ntl.sum(ntl.where(x > 0, x + x, 0.0), axis=1)  # noqa: F841


def _atomic_offset(x, out):
    ntl.atomic_add(out.source.data_ptr() + 1, x)
    ntl.atomic_add(out.source.data_ptr(), 0.0)


def _add(x, y, out):
    out = x + y  # noqa: F841


def _row_arrangement(nt_max, stream):
    return nt_max.tile((1, -1)), stream.tile((1,))


def _row_maximum(nt_max, stream):
    stream = ntl.max(nt_max, axis=1)  # noqa: F841


def _window_arrangement(x, out):
    x = x.pad(((0, 0), (0, 0), (1, 1)))
    x = x.tile((1, 1, 18), strides=(-1, -1, 16), floor_mode=True)
    x = x.ravel().flatten(end_dim=3).flatten(start_dim=1).tile((1, -1))
    x.dtype = x.dtype.squeeze(0)
    out = out.tile((1, 1, 32), strides=(-1, -1, 32), floor_mode=True)
    out = out.ravel().flatten(end_dim=3).flatten(start_dim=1).tile((1, -1))
    out.dtype = out.dtype.squeeze(0)

    return x, out


def _window_neighbors(x, out):
    for i in range(16):
        out[2 * i] = x[i]
        out[2 * i + 1] = x[i + 2]


def test_cuda_native_numeric_semantics():
    extrema = nt.make(
        _arrangement,
        _extrema,
        tuple(nt.Tensor(1) for _ in range(4)),
        backend="cuda",
        max_num_configs=1,
    )

    for dtype, values in (
        (torch.float32, [float("nan"), 2, -3, float("inf"), -float("inf")]),
        (torch.int64, [(1 << 54) + 1, (1 << 54) + 3, -(1 << 54) - 1, 2, -3]),
    ):
        x = torch.tensor(values, dtype=dtype, device="cuda").repeat(52)[:257]
        y = x.roll(1)
        maximum, minimum = torch.empty_like(x), torch.empty_like(x)
        extrema(x, y, maximum, minimum)
        torch.testing.assert_close(
            maximum, torch.maximum(x, y), rtol=0, atol=0, equal_nan=True
        )
        torch.testing.assert_close(
            minimum, torch.minimum(x, y), rtol=0, atol=0, equal_nan=True
        )

    for dtype in (torch.float16, torch.bfloat16):
        kernel = nt.make(
            _arrangement,
            _rounded_arithmetic,
            tuple(nt.Tensor(1, dtype=str(dtype)) for _ in range(4)),
            backend="cuda",
            max_num_configs=1,
        )
        x, y, z = (torch.randn(257, dtype=dtype, device="cuda") for _ in range(3))
        out = torch.empty_like(x)
        kernel(x, y, z, out)
        torch.testing.assert_close(out, (x + y) * z, atol=0, rtol=0)

        atomic = nt.make(
            _arrangement,
            _atomic_offset,
            (nt.Tensor(1, dtype=str(dtype)), nt.Tensor(1, dtype=str(dtype))),
            backend="cuda",
            max_num_configs=1,
        )
        values = torch.ones(4, device="cuda", dtype=dtype)
        totals = torch.zeros(2, device="cuda", dtype=dtype)
        atomic(values, totals)
        torch.testing.assert_close(
            totals, torch.tensor([0, 4], device="cuda", dtype=dtype)
        )

        selected_sum = nt.make(
            _row_arrangement,
            _positive_sum,
            (nt.Tensor(2, dtype=str(dtype)), nt.Tensor(1, dtype=str(dtype))),
            backend="cuda",
            max_num_configs=1,
        )
        x = ((torch.arange(7 * 127, device="cuda") % 13 - 6) / 4).to(dtype)
        x = x.reshape(7, 127)
        out = torch.empty(7, device="cuda", dtype=dtype)
        selected_sum(x, out)
        torch.testing.assert_close(
            out, torch.where(x > 0, x + x, 0.0).sum(1), rtol=0, atol=0
        )

    reduction = nt.make(
        _row_arrangement,
        _row_maximum,
        (nt.Tensor(2), nt.Tensor(1)),
        backend="cuda",
        max_num_configs=1,
        pass_options={
            "ssa.cuda.optimize_schedule": {"candidate": "cooperative-reduction-128"}
        },
    )
    x = torch.randn(17, 127, device="cuda")
    x[0, 42] = float("nan")
    out = torch.empty(17, device="cuda")
    reduction(x, out)
    torch.testing.assert_close(out, x.amax(1), rtol=0, atol=0, equal_nan=True)

    window = nt.make(
        _window_arrangement,
        _window_neighbors,
        (nt.Tensor(3, other=0.0), nt.Tensor(3)),
        backend="cuda",
        max_num_configs=1,
    )
    x = torch.arange(192, device="cuda", dtype=torch.float32).reshape(2, 3, 32)
    out = torch.empty((2, 3, 64), device="cuda")
    window(x, out)
    torch.cuda.synchronize()
    expected = torch.empty_like(out)
    expected[..., ::2] = torch.nn.functional.pad(x[..., :-1], (1, 0))
    expected[..., 1::2] = torch.nn.functional.pad(x[..., 1:], (0, 1))
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_cuda_native_launch_revalidates_storage_and_current_stream():
    kernel = nt.make(
        _arrangement,
        _add,
        tuple(nt.Tensor(1, dtype=nt.float32) for _ in range(3)),
        backend="cuda",
        max_num_configs=1,
    )
    x, y = (torch.randn(257, device="cuda") for _ in range(2))
    out = torch.empty_like(x)
    kernel(x, y, out)
    torch.testing.assert_close(out, x + y)

    original_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(original_stream)

    with torch.cuda.stream(stream):
        x.fill_(3)
        kernel(x, y, out)
        completed = torch.cuda.Event()
        completed.record()

    original_stream.wait_event(completed)
    torch.testing.assert_close(out, x + y)

    storage = torch.full((259,), -123.0, device="cuda")
    out.set_(storage[1:-1])
    kernel(x, y, out)
    torch.testing.assert_close(out, x + y)
    assert storage[0].item() == storage[-1].item() == -123

    owner = weakref.ref(out)
    del out
    gc.collect()
    assert owner() is None


def _divide_arrangement(x, out, scale):
    return x.tile((128,)), out.tile((128,)), scale


def _divide_by_scalar(x, out, scale):
    out = x / scale  # noqa: F841


def test_cuda_native_cached_scalar_preserves_representation():
    kernel = nt.make(
        _divide_arrangement,
        _divide_by_scalar,
        (
            nt.Tensor(1, dtype=nt.float32),
            nt.Tensor(1, dtype=nt.float32),
            nt.Tensor(0, dtype=nt.float32),
        ),
        backend="cuda",
        max_num_configs=1,
    )
    x = torch.ones(257, device="cuda")
    out = torch.empty_like(x)
    scalar = torch.tensor(0.0, device="cuda")

    # Generic Python scalar, tensor item guard, and keyword two-tensor fast path.
    for path in ("positional", "tensor", "keyword"):
        for value in (0.0, -0.0, 0.0, 7.0):
            scalar.fill_(value)

            if path == "keyword":
                kernel(x, out, scale=value)
            else:
                kernel(x, out, scalar if path == "tensor" else value)

            expected = x / scalar
            torch.testing.assert_close(out, expected)
            assert torch.equal(torch.signbit(out), torch.signbit(expected))
