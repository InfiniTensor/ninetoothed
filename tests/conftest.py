import hashlib
import random

import pytest
import torch


def pytest_collectstart(collector):
    if isinstance(collector, pytest.Module):
        _set_random_seed(_hash(collector.name))


@pytest.fixture(scope="module", autouse=True)
def set_seed_per_module(request):
    _set_random_seed(_hash(_module_path_from_request(request)))


@pytest.fixture(autouse=True)
def set_seed_per_test(request):
    _set_random_seed(_hash(_test_case_path_from_request(request)))


def _set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def _test_case_path_from_request(request):
    return f"{_module_path_from_request(request)}::{request.node.name}"


def _module_path_from_request(request):
    return f"{request.module.__name__.replace('.', '/')}.py"


def _hash(string):
    return int(hashlib.sha256(string.encode("utf-8")).hexdigest(), 16) % 2**32


@pytest.fixture(autouse=True)
def select_device_backend(request, monkeypatch):
    """Route generic NPU numerical tests through the Ascend backend."""
    if getattr(request.node, "callspec", None) is not None:
        if request.node.callspec.params.get("device") == "npu":
            monkeypatch.setenv("NINETOOTHED_BACKEND", "ascend")
            name = (
                torch.npu.get_device_name(0)
                .lower()
                .removeprefix("ascend")
                .split("-")[0]
            )
            monkeypatch.setenv("NINETOOTHED_PLATFORM", f"ascend-{name}")
