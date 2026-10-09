from types import SimpleNamespace

import pytest

from tests import utils


@pytest.mark.parametrize(
    "available, backend, expected",
    (
        ((), None, ()),
        (("npu",), None, ("npu",)),
        (("npu",), "triton", ()),
        (("cuda", "mlu", "npu"), None, ("cuda", "mlu", "npu")),
        (("cuda", "mlu", "npu"), "triton", ("cuda", "mlu")),
        (("cuda", "mlu", "npu"), "ascend", ("npu",)),
    ),
)
def test_device_discovery_respects_backend(monkeypatch, available, backend, expected):
    runtimes = {
        device: SimpleNamespace(is_available=lambda device=device: device in available)
        for device in ("cuda", "mlu", "npu")
    }
    monkeypatch.setattr(utils, "torch", SimpleNamespace(**runtimes))

    assert utils.get_available_devices(backend=backend) == expected


def test_optional_runtime_packages_are_not_required(monkeypatch):
    monkeypatch.setattr(
        utils,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    assert utils.get_available_devices() == ()
