import importlib

import pytest
import torch

import ninetoothed
from tests.utils import get_available_devices

parallelize_module = importlib.import_module("ninetoothed.parallelize")


def _allocate_or_skip(factory, *args, **kwargs):
    try:
        return factory(*args, **kwargs)
    except (torch.AcceleratorError, torch.OutOfMemoryError) as exc:
        pytest.skip(f"Could not allocate test tensors on the device: {exc}")


def test_parallelize_recognizes_add(monkeypatch):
    captured = {}

    def fake_make(arrangement, application, tensors, **kwargs):
        captured["arrangement"] = arrangement
        captured["application"] = application
        captured["tensors"] = tensors
        captured["kwargs"] = kwargs

        def kernel(*args, **inner_kwargs):
            captured["args"] = args
            captured["inner_kwargs"] = inner_kwargs

        return kernel

    monkeypatch.setattr(parallelize_module, "make", fake_make)

    @ninetoothed.parallelize
    def add(input, other, output):
        for i in range(output.shape[0]):
            output[i] = input[i] + other[i]

    add("input", "other", "output")

    assert add._serial_kind == "add"
    assert captured["kwargs"]["kernel_name"] == "add"
    assert len(captured["tensors"]) == 3
    assert captured["args"] == ("input", "other", "output")


def test_parallelize_recognizes_mm(monkeypatch):
    captured = {}

    def fake_make(arrangement, application, tensors, **kwargs):
        captured["arrangement"] = arrangement
        captured["application"] = application
        captured["tensors"] = tensors
        captured["kwargs"] = kwargs

        def kernel(*args, **inner_kwargs):
            captured["args"] = args
            captured["inner_kwargs"] = inner_kwargs

        return kernel

    monkeypatch.setattr(parallelize_module, "make", fake_make)

    @ninetoothed.parallelize
    def mm(input, other, output):
        for m in range(output.shape[0]):
            for n in range(output.shape[1]):
                acc = 0.0
                for k in range(input.shape[1]):
                    acc += input[m, k] * other[k, n]
                output[m, n] = acc

    mm("input", "other", "output")

    assert mm._serial_kind == "mm"
    assert captured["kwargs"]["kernel_name"] == "mm"
    assert len(captured["tensors"]) == 3
    assert captured["args"] == ("input", "other", "output")


def test_parallelize_rejects_unsupported_serial_kernel():
    with pytest.raises(NotImplementedError, match="supported serial kernel"):

        @ninetoothed.parallelize
        def square(input, output):
            for i in range(output.shape[0]):
                output[i] = input[i] * input[i]


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("size", (8192,))
def test_parallelize_add(device, dtype, size):
    @ninetoothed.parallelize
    def add(input, other, output):
        for i in range(output.shape[0]):
            output[i] = input[i] + other[i]

    input = _allocate_or_skip(torch.rand, size, dtype=dtype, device=device)
    other = _allocate_or_skip(torch.rand, size, dtype=dtype, device=device)
    output = _allocate_or_skip(torch.empty_like, input)

    add(input, other, output)

    assert torch.allclose(output, input + other)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("size", (128,))
def test_parallelize_mm(device, size):
    @ninetoothed.parallelize
    def mm(input, other, output):
        for m in range(output.shape[0]):
            for n in range(output.shape[1]):
                acc = 0.0
                for k in range(input.shape[1]):
                    acc += input[m, k] * other[k, n]
                output[m, n] = acc

    input = _allocate_or_skip(
        torch.randn, (size, size), dtype=torch.float16, device=device
    )
    other = _allocate_or_skip(
        torch.randn, (size, size), dtype=torch.float16, device=device
    )
    output = _allocate_or_skip(
        torch.empty, (size, size), dtype=torch.float16, device=device
    )

    mm(input, other, output)

    assert torch.allclose(output, torch.matmul(input, other), atol=0.15)
