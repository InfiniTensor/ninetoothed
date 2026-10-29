import gc
import weakref

import pytest
import torch

import ninetoothed as nt
import ninetoothed.language as ntl


def _arrangement(*tensors):
    return tuple(tensor.tile((128,)) for tensor in tensors)


def _extrema(nt_max, nt_min, stream, minimum):
    stream = ntl.maximum(nt_max, nt_min)  # noqa: F841
    minimum = ntl.minimum(nt_max, nt_min)  # noqa: F841


def _rounded_arithmetic(x, y, z, out):
    out = (x + y) * z  # noqa: F841


def _add(x, y, out):
    out = x + y  # noqa: F841


def _row_arrangement(nt_max, stream):
    return nt_max.tile((1, -1)), stream.tile((1,))


def _row_maximum(nt_max, stream):
    stream = ntl.max(nt_max, axis=1)  # noqa: F841


def _scalar_arrangement(x, out, scale):
    return x, out.tile((128,)), scale


def _scaled_scalar(x, out, scale):
    out = x * scale  # noqa: F841


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
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

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

    x = torch.tensor([0.0, -0.0], device="cuda")
    y = -x
    maximum, minimum = torch.empty_like(x), torch.empty_like(x)
    extrema(x, y, maximum, minimum)
    assert torch.equal(torch.signbit(maximum), torch.signbit(torch.maximum(x, y)))
    assert torch.equal(torch.signbit(minimum), torch.signbit(torch.minimum(x, y)))

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
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

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

    with pytest.raises(TypeError, match="dtype"):
        kernel(x.half(), y, out)

    with pytest.raises(TypeError, match="CUDA device"):
        kernel(x.cpu(), y, out)

    owner = weakref.ref(out)
    del out
    gc.collect()
    assert owner() is None

    scaled = nt.make(
        _scalar_arrangement,
        _scaled_scalar,
        (
            nt.Tensor(0, dtype=nt.float32),
            nt.Tensor(1, dtype=nt.float32),
            nt.Tensor(0, dtype=nt.float32),
        ),
        backend="cuda",
        max_num_configs=1,
    )
    scalar = torch.tensor(3.0, device="cuda")
    out = torch.empty(257, device="cuda")

    for value in (3.0, 7.0, -2.0):
        scalar.fill_(value)
        scaled(scalar, out, scale=2.0)
        torch.testing.assert_close(out, torch.full_like(out, value * 2))
