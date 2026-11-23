import sys
from pathlib import Path

import pytest
import torch

# Add the skill root to sys.path so examples/ can be imported
SKILL_ROOT = Path(__file__).resolve().parent.parent
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "benchmark: performance benchmarks (deselected by default)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip benchmark tests unless explicitly selected via ``-m benchmark``."""
    if config.getoption("-m") == "benchmark":
        return

    skip = pytest.mark.skip(reason="benchmarks are not selected, use `-m benchmark`")

    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)

DTYPE = torch.float16
DEVICE = "cuda"


@pytest.fixture(autouse=True)
def seed():
    """Set a fixed random seed before each test for reproducibility."""
    torch.manual_seed(0)


@pytest.fixture
def dtype():
    return DTYPE


@pytest.fixture
def device():
    return DEVICE


def assert_allclose(actual, expected, atol=1e-3, rtol=1e-3):
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"Max diff: {(actual - expected).abs().max().item():.6f} "
        f"(atol={atol}, rtol={rtol})"
    )
