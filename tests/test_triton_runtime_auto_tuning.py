import uuid

import pytest
import torch

import ninetoothed
import ninetoothed.auto_tuner as auto_tuner
from ninetoothed import Tensor
from tests.utils import get_available_devices


def _arrangement(input, other, output):
    return tuple(tensor.tile((64,)) for tensor in (input, other, output))


def _application(input, other, output):
    output = input + other  # noqa: F841


@pytest.mark.parametrize("device", get_available_devices())
def test_triton_tuple_configurations_are_benchmarked_and_cached(device, monkeypatch):
    benchmarked = []

    def benchmark(function, args, kwargs):
        benchmarked.append(function)
        function(*args, **kwargs)

        return 2.0 if len(benchmarked) == 1 else 1.0

    monkeypatch.setattr(auto_tuner, "_default_benchmark", benchmark)
    handle = ninetoothed.make(
        _arrangement,
        _application,
        tuple(Tensor(shape=(257,), dtype=ninetoothed.float32) for _ in range(3)),
        backend="triton",
        kernel_name=f"runtime_auto_tuning_{uuid.uuid4().hex}",
        num_warps=(4, 8),
        num_stages=(1,),
        max_num_configs=2,
    )
    input = torch.randn(257, device=device)
    other = torch.randn_like(input)
    output = torch.empty_like(input)

    handle(input, other, output)
    assert torch.allclose(output, input + other)
    assert len(benchmarked) == 2

    arg_key = handle._tuner._make_arg_key((input, other, output), {})
    assert handle._tuner._best_func[arg_key] is benchmarked[1]
    assert handle._selected_tuning_candidate["id"] == "warps-8_stages-1"

    handle(input, other, output)
    assert len(benchmarked) == 2


def test_auto_tuner_validates_arguments_before_benchmark(tmp_path, monkeypatch):
    monkeypatch.setattr(auto_tuner, "_AUTO_TUNING_CACHE_DIR", tmp_path)
    benchmarked = []

    def validate(args, kwargs):
        del args, kwargs

        raise TypeError("Invalid public arguments.")

    tuner = auto_tuner.AutoTuner(
        (lambda value: value, lambda value: value),
        ("first", "second"),
        benchmark=lambda function, args, kwargs: benchmarked.append(function),
        cache_namespace="validation",
        validator=validate,
    )

    with pytest.raises(TypeError, match="Invalid public arguments"):
        tuner(1)

    assert not benchmarked


def test_auto_tuner_cache_is_reused_across_instances(tmp_path, monkeypatch):
    monkeypatch.setattr(auto_tuner, "_AUTO_TUNING_CACHE_DIR", tmp_path)
    benchmarked = []

    def benchmark(function, args, kwargs):
        del args, kwargs
        benchmarked.append(function)

        return float(len(benchmarked))

    first = auto_tuner.AutoTuner(
        (lambda value: value + 1, lambda value: value + 2),
        ("first", "second"),
        benchmark=benchmark,
        cache_namespace="shared",
    )
    assert first(1) == 2
    assert len(benchmarked) == 2

    second = auto_tuner.AutoTuner(
        (lambda value: value + 3, lambda value: value + 4),
        ("first", "second"),
        benchmark=benchmark,
        cache_namespace="shared",
    )
    assert second(1) == 4
    assert len(benchmarked) == 2


def test_auto_tuner_retries_transient_candidate_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(auto_tuner, "_AUTO_TUNING_CACHE_DIR", tmp_path)

    def fail(function, args, kwargs):
        del function, args, kwargs

        raise RuntimeError("Transient compiler failure.")

    first = auto_tuner.AutoTuner(
        (lambda value: value + 1, lambda value: value + 2),
        ("first", "second"),
        benchmark=fail,
        cache_namespace="retry",
    )

    with pytest.raises(RuntimeError, match="All auto-tuning candidates failed"):
        first(1)

    benchmarked = []

    def succeed(function, args, kwargs):
        benchmarked.append(function)
        function(*args, **kwargs)

        return float(len(benchmarked))

    second = auto_tuner.AutoTuner(
        (lambda value: value + 3, lambda value: value + 4),
        ("first", "second"),
        benchmark=succeed,
        cache_namespace="retry",
    )
    assert second(1) == 4
    assert len(benchmarked) == 2
