import hashlib
import os
import random

import pytest
import torch

# Tests that do not pin a backend inherit NINETOOTHED_BACKEND.  When the
# caller has not pinned one, prefer a backend whose full toolchain actually
# runs on this host (e.g. bangc inside a Cambricon-only container) so that
# plain `pytest` invocations do not fail on unrelated platforms.
if not os.environ.get("NINETOOTHED_BACKEND"):
    from tests.utils import detect_default_backend

    os.environ["NINETOOTHED_BACKEND"] = detect_default_backend()

# Known capability limits of specific backends, keyed by the effective
# backend.  Each entry is a substring matched against the test node id plus
# a skip reason.  This is the interim single place for backend gaps until
# the planned multi-platform test restructuring lands.
_BACKEND_TEST_LIMITATIONS = {
    "bangc": (
        (
            "test_attention",
            "flash-attention long sequences fall back to the O(n^2) BangC "
            "lowering and exceed the MLU kernel timeout",
        ),
    ),
}


def pytest_collection_modifyitems(config, items):
    backend = os.environ.get("NINETOOTHED_BACKEND", "triton")
    limitations = _BACKEND_TEST_LIMITATIONS.get(backend, ())

    if not limitations:
        return

    for item in items:
        for marker, reason in limitations:
            if marker in item.nodeid:
                item.add_marker(pytest.mark.skip(reason=reason))


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
