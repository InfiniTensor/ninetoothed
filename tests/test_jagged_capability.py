import pytest
import torch

from tests.capabilities import CapabilityResult, jagged
from tests.test_jagged import (
    _expanded_values_from_batches,
    _padded_from_batches,
)


@pytest.fixture(autouse=True)
def clear_jagged_probe_caches():
    probes = (jagged.jagged_dim_1, jagged.jagged_dim_2)

    for probe in probes:
        probe.cache_clear()

    yield

    for probe in probes:
        probe.cache_clear()


def test_jagged_layout_probes_are_distinct_and_cached(monkeypatch):
    calls = []

    def fake_run(name, source, *, timeout=60):
        calls.append((name, source, timeout))

        return CapabilityResult(True)

    monkeypatch.setattr(jagged, "run_python_probe", fake_run)
    jagged.jagged_dim_1()
    jagged.jagged_dim_1()
    jagged.jagged_dim_2()

    assert [name for name, _source, _timeout in calls] == [
        "torch jagged_dim=1",
        "torch jagged_dim=2",
    ]
    assert all("layout=torch.jagged" in source for _name, source, _timeout in calls)


def test_padded_oracle_uses_original_batches():
    batches = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0, 6.0]]),
    )

    actual = _padded_from_batches(batches, padding=-1, jagged_dim=1)
    expected = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [-1.0, -1.0]],
        ]
    )

    torch.testing.assert_close(actual, expected)


def test_padded_oracle_supports_second_jagged_dimension():
    batches = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0], [6.0]]),
    )

    actual = _padded_from_batches(batches, padding=0, jagged_dim=2)
    expected = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 0.0], [6.0, 0.0]],
        ]
    )

    torch.testing.assert_close(actual, expected)


def test_expanded_values_oracle_uses_batch_lengths():
    batches = (
        torch.empty((2, 2)),
        torch.empty((1, 2)),
    )
    source = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])

    actual = _expanded_values_from_batches(batches, source, jagged_dim=1)
    expected = torch.cat((source[0].expand(2, 2), source[1].expand(1, 2)))

    torch.testing.assert_close(actual, expected)


def test_expanded_values_oracle_supports_second_jagged_dimension():
    batches = (
        torch.empty((2, 2)),
        torch.empty((2, 1)),
    )
    source = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])

    actual = _expanded_values_from_batches(batches, source, jagged_dim=2)
    expected = torch.tensor([[1.0, 1.0, 3.0], [2.0, 2.0, 4.0]])

    torch.testing.assert_close(actual, expected)
