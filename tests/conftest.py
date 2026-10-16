import hashlib
import random

import pytest

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

    torch = None


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

    if torch is not None:
        torch.manual_seed(seed)


def _test_case_path_from_request(request):
    return f"{_module_path_from_request(request)}::{request.node.name}"


def _module_path_from_request(request):
    return f"{request.module.__name__.replace('.', '/')}.py"


def _hash(string):
    return int(hashlib.sha256(string.encode("utf-8")).hexdigest(), 16) % 2**32
