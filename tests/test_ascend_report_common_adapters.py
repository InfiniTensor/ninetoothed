import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.ascend_report import operator_adapters as adapters


class Tensor:
    def __init__(self, values, events):
        self.values = np.asarray(values)
        self.events = events
        self.shape = self.values.shape
        self.ndim = self.values.ndim
        self.dtype = self.values.dtype
        self.device = "npu"

    def reshape(self, *shape):
        self.events.append(("reshape", self.shape))

        return Tensor(self.values.reshape(*shape), self.events)


def common_case(operator):
    original = f"{operator}__float32__test"

    return {
        "id": original + "__common_v1",
        "original_id": original,
        "operator": operator,
        "case_profile": "common_contracts_v1",
    }


@pytest.fixture
def runtime(monkeypatch):
    events = []
    modules = {}

    def empty(shape, *, dtype, device):
        assert device == "npu"
        events.append(("allocate", tuple(shape)))

        return Tensor(np.empty(shape, dtype=dtype), events)

    def load(name):
        events.append(("import", name))

        return modules[name]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(empty=empty))
    monkeypatch.setattr(adapters, "_module", load)

    return SimpleNamespace(events=events, modules=modules)


@pytest.mark.parametrize("operator", ("clamp", "dropout", "max_pool2d"))
@pytest.mark.parametrize("profile", (None, "original", "common_contracts_v2"))
def test_original_contracts_still_reject_before_importing_kernels(
    monkeypatch, operator, profile
):
    case = {"operator": operator, "case_profile": profile}
    monkeypatch.setitem(sys.modules, "torch", None)

    def unexpected_import(name):
        raise AssertionError(f"Unsupported case imported {name}")

    monkeypatch.setattr(adapters, "_module", unexpected_import)

    with pytest.raises(adapters.UnsupportedCase):
        adapters.build_call(case, {"args": ()})


def test_common_clamp_reshapes_inside_each_call_and_uses_original_kernel(runtime):
    values = np.array([[-2.0, 0.2, 3.0], [-0.4, 1.5, -1.5]], dtype=np.float32).T
    x = Tensor(values, runtime.events)
    launches = []

    def kernel(flat, out):
        assert flat.ndim == out.ndim == 1
        launches.append((flat, out))
        out.values[:] = np.clip(flat.values, -1, 1)

    runtime.modules["pointwise.clamp"] = SimpleNamespace(kernel=kernel)
    case = common_case("clamp")
    call, metadata = adapters.build_call(
        case, {"args": (x,), "parameters": {"min": -1.0, "max": 1.0}}
    )
    assert runtime.events == [("import", "pointwise.clamp")]
    assert metadata["case_profile"] == "common_contracts_v1"
    assert metadata["original_case_id"] == case["original_id"]
    assert metadata["numeric_scope"] == "clamp_fixed_minus_one_to_one"
    assert metadata["source_modules"] == ["ntops_lab.kernels.pointwise.clamp"]

    for scale in (1, -2):
        x.values[:] *= scale
        output = call()
        np.testing.assert_array_equal(output.values, np.clip(x.values, -1, 1))
        assert output.shape == x.shape
        assert np.shares_memory(output.values, launches[-1][1].values)

    assert len(launches) == 2
    assert sum(event[0] == "allocate" for event in runtime.events) == 2
    assert sum(event == ("reshape", x.shape) for event in runtime.events) == 2


@pytest.mark.parametrize(
    "parameters",
    (
        {"min": -0.25, "max": 0.75},
        {"min": -1, "max": True},
        {"min": -1},
        {"min": -1, "max": 1, "inplace": False},
    ),
)
def test_common_clamp_rejects_nonmatching_parameters_before_dispatch(
    runtime, parameters
):
    with pytest.raises(adapters.UnsupportedCase) as error:
        adapters.build_call(
            common_case("clamp"), {"args": (), "parameters": parameters}
        )

    assert error.value.category == "parameter_contract"
    assert runtime.events == []


@pytest.mark.parametrize("shape", ((7,), (2, 3)))
@pytest.mark.parametrize(
    "probability,training", ((0.25, False), (1.0, False), (0, True))
)
def test_common_dropout_calls_existing_identity_run_and_restores_shape(
    runtime, shape, probability, training
):
    x = Tensor(
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape), runtime.events
    )
    launches = []

    def run(rows, **kwargs):
        assert rows.ndim == 2
        output = Tensor(rows.values.copy(), runtime.events)
        launches.append((rows, kwargs, output))

        return output

    runtime.modules["pointwise.dropout"] = SimpleNamespace(run=run)
    call, metadata = adapters.build_call(
        common_case("dropout"),
        {
            "args": (x,),
            "parameters": {
                "p": probability,
                "training": training,
                "return_mask": False,
            },
        },
    )
    assert runtime.events == [("import", "pointwise.dropout")]
    output = call()
    rows, keywords, kernel_output = launches[0]
    assert keywords == {"p": probability, "training": training, "inplace": False}
    assert rows.shape == ((1, 7) if len(shape) == 1 else shape)
    assert output.shape == shape
    np.testing.assert_array_equal(output.values, x.values)
    assert np.shares_memory(output.values, kernel_output.values)
    assert not np.shares_memory(output.values, x.values)
    assert metadata["output_contract"] == "tensor"
    assert metadata["numeric_scope"] == (
        "deterministic_identity_p0" if training else "inference_identity"
    )
    assert metadata["source_modules"] == ["ntops_lab.kernels.pointwise.dropout"]
    assert len(launches) == 1


@pytest.mark.parametrize(
    "parameters",
    (
        {"p": 0.25, "training": True, "return_mask": False},
        {"p": 0, "training": False, "return_mask": True},
        {"p": 0, "training": False, "seed": 42},
        {"p": 0, "training": False, "offset": 0},
        {"p": 0, "training": False, "inplace": False},
        {"p": -0.1, "training": False},
        {"p": 1.1, "training": False},
        {"p": float("nan"), "training": False},
        {"p": float("inf"), "training": False},
        {"p": True, "training": False},
        {"p": 0, "training": "False"},
        {"p": 0, "training": False, "return_mask": 0},
    ),
)
def test_common_dropout_rejects_stochastic_mask_and_unknown_contracts(
    runtime, parameters
):
    with pytest.raises(adapters.UnsupportedCase) as error:
        adapters.build_call(
            common_case("dropout"), {"args": (), "parameters": parameters}
        )

    assert error.value.category == "parameter_contract"
    assert runtime.events == []


@pytest.mark.parametrize("shape", ((), (2, 3, 4)))
def test_common_dropout_rejects_ranks_without_a_source_contract(runtime, shape):
    x = Tensor(np.zeros(shape), runtime.events)

    with pytest.raises(adapters.UnsupportedCase, match="rank-one or rank-two"):
        adapters.build_call(
            common_case("dropout"),
            {"args": (x,), "parameters": {"p": 0.25, "training": False}},
        )

    assert runtime.events == []


@pytest.mark.parametrize("paired", (False, True))
def test_common_max_pool_dispatches_values_only_existing_helper(runtime, paired):
    x = Tensor(np.zeros((1, 2, 7, 5), dtype=np.float32), runtime.events)
    expected = Tensor(np.ones((1, 2, 4, 3), dtype=np.float32), runtime.events)
    calls = []

    def pool(input_, kind, **kwargs):
        calls.append((input_, kind, kwargs))

        return expected

    runtime.modules["layout.avg_pool2d"] = SimpleNamespace(_pool2d=pool)
    parameters = {
        "kernel_size": [3, 3] if paired else 3,
        "stride": (2, 2) if paired else 2,
        "padding": [1, 1] if paired else 1,
        "return_indices": False,
    }
    call, metadata = adapters.build_call(
        common_case("max_pool2d"), {"args": (x,), "parameters": parameters}
    )
    assert not calls
    assert call() is expected
    assert calls == [
        (x, "max", {"kernel_size": (3, 3), "stride": (2, 2), "padding": (1, 1)})
    ]
    assert metadata["output_contract"] == "tensor"
    assert metadata["numeric_scope"] == "max_pool2d_values_only"
    assert metadata["source_modules"] == ["ntops_lab.kernels.layout.avg_pool2d"]


@pytest.mark.parametrize(
    "override",
    (
        {"kernel_size": 4},
        {"kernel_size": 3.0},
        {"stride": 1},
        {"padding": 0},
        {"padding": True},
        {"return_indices": True},
        {"return_indices": 0},
        {"dilation": 1},
        {"ceil_mode": False},
    ),
)
def test_common_max_pool_rejects_different_windows_and_index_requests(
    runtime, override
):
    parameters = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "return_indices": False,
        **override,
    }

    with pytest.raises(adapters.UnsupportedCase) as error:
        adapters.build_call(
            common_case("max_pool2d"), {"args": (), "parameters": parameters}
        )

    assert error.value.category == "parameter_contract"
    assert runtime.events == []


def test_common_profile_keeps_existing_addmm_parameter_boundary(runtime):
    with pytest.raises(adapters.UnsupportedCase, match="alpha=beta=1"):
        adapters.build_call(
            common_case("addmm"),
            {"args": (), "parameters": {"alpha": 0.5, "beta": 1}},
        )

    assert runtime.events == []
