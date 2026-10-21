import collections
import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.ascend_report import e2e_linear


class _Tensor:
    def __init__(self, values, device=None):
        self.values = np.asarray(values)
        self.device = device or SimpleNamespace(type="npu", index=0)

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def dtype(self):
        return self.values.dtype

    def stride(self):
        return tuple(value // self.values.itemsize for value in self.values.strides)

    def reshape(self, *shape):
        return _Tensor(self.values.reshape(*shape), self.device)

    def view(self, shape):
        return self.reshape(shape)

    def transpose(self, first, second):
        return _Tensor(self.values.swapaxes(first, second), self.device)

    def contiguous(self):
        raise AssertionError("Linear adapter must not copy model weights.")


class _Linear:
    def __init__(self, weight, bias=None):
        self.weight = weight
        self.bias = bias

    def forward(self, input):
        raise AssertionError("Native Linear fallback must not be called.")

    def __call__(self, input):
        return self.forward(input)


@pytest.fixture
def harness(monkeypatch):
    state = SimpleNamespace(calls=[], disabled=0, device_contexts=0)

    @contextlib.contextmanager
    def disable_dispatch():
        state.disabled += 1
        try:
            yield
        finally:
            state.disabled -= 1

    @contextlib.contextmanager
    def device_context(device):
        assert device.type == "npu"
        state.device_contexts += 1
        yield

    def generated(left, right, output):
        assert state.disabled
        state.calls.append(("ntops", left, right, output))
        output.values[...] = (left.values @ right.values).astype(np.float16)

    class BaselineKernel:
        def __getitem__(self, grid):
            def launch(*args):
                assert state.disabled
                state.calls.append(
                    ("triton", args[0], args[1], args[2], grid, args[3:])
                )
                args[2].values[...] = args[0].values @ args[1].values

            return launch

    def fail_public_run(*args):
        raise AssertionError("The public NPU wrapper uses a forbidden fallback.")

    module = SimpleNamespace(
        __name__="ntops_lab.kernels.linear.mm",
        __file__="synthetic/ntops/mm.py",
        kernel=generated,
        run=fail_public_run,
    )
    baseline_module = SimpleNamespace(
        __name__="triton_ascend_ops.kernels.mm",
        __file__="synthetic/baseline/mm.py",
        mm_kernel=BaselineKernel(),
    )
    imports = {
        "ntops_lab.kernels.linear.mm": module,
        "triton_ascend_ops": SimpleNamespace(
            load_operator=lambda name: baseline_module
        ),
        "triton.language": SimpleNamespace(float32="tl.float32"),
    }
    monkeypatch.setattr(
        e2e_linear, "importlib", SimpleNamespace(import_module=imports.__getitem__)
    )
    state.torch = SimpleNamespace(
        float16=np.dtype("float16"),
        bfloat16="bfloat16",
        float32=np.dtype("float32"),
        nn=SimpleNamespace(Linear=_Linear),
        _C=SimpleNamespace(_DisableTorchDispatch=disable_dispatch),
        npu=SimpleNamespace(device=device_context),
        empty=lambda shape, device, dtype: _Tensor(
            np.empty(shape, dtype=dtype), device
        ),
    )
    state.recorder = SimpleNamespace(
        collect=True,
        calls=collections.Counter(),
        shapes=collections.Counter(),
        source_imports={},
        active_operation="unclassified",
    )

    return state


@pytest.mark.parametrize("backend", ("ntops", "triton"))
@pytest.mark.parametrize("shape", ((5,), (1, 1, 5), (1, 19, 5), (2, 3, 5)))
def test_linear_restores_leading_shape_and_uses_transposed_weight_view(
    harness, backend, shape
):
    adapter = e2e_linear.Adapter(backend, harness.torch, recorder=harness.recorder)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 11
    weight = _Tensor(np.arange(35, dtype=np.float32).reshape(7, 5) / 13)
    input = _Tensor(values, weight.device)
    output = adapter.linear(
        input, weight, module_name="model.layers.0.self_attn.q_proj"
    )
    expected = input.values @ weight.values.T

    if backend == "ntops":
        expected = expected.astype(np.float16).astype(np.float32)

    np.testing.assert_array_equal(output.values, expected)
    assert output.shape == (*shape[:-1], 7)
    dispatch = harness.calls[-1]
    assert dispatch[0] == backend
    assert np.shares_memory(dispatch[2].values, weight.values)
    assert dispatch[2].stride() == (1, 5)
    assert np.shares_memory(dispatch[3].values, output.values)
    assert harness.recorder.calls == {"linear:q_proj": 1}
    assert harness.recorder.active_operation == "linear_mm"
    assert harness.recorder.source_imports


def test_baseline_launch_preserves_the_existing_kernel_signature(harness):
    adapter = e2e_linear.Adapter("triton", harness.torch)
    weight = _Tensor(np.ones((33, 5), dtype=np.float32))
    input = _Tensor(np.ones((1, 19, 5), dtype=np.float32), weight.device)
    adapter.linear(input, weight)
    _, left, right, out, grid, arguments = harness.calls[-1]

    assert grid == (2, 1)
    assert arguments == (
        19,
        33,
        5,
        *left.stride(),
        *right.stride(),
        *out.stride(),
        "tl.float32",
        32,
        32,
        32,
        1,
        1,
        False,
    )
    assert harness.device_contexts == 1


@pytest.mark.parametrize("backend", ("ntops", "triton"))
def test_noncontiguous_activation_is_reshaped_symmetrically(harness, backend):
    adapter = e2e_linear.Adapter(backend, harness.torch)
    weight = _Tensor(np.ones((3, 5), dtype=np.float32))
    input = _Tensor(
        np.arange(30, dtype=np.float32).reshape(3, 2, 5).swapaxes(0, 1), weight.device
    )
    output = adapter.linear(input, weight)

    np.testing.assert_array_equal(output.values, input.values @ weight.values.T)
    assert np.shares_memory(harness.calls[-1][2].values, weight.values)


def test_patch_records_all_projection_classes_and_restores_after_exception(harness):
    adapter = e2e_linear.Adapter("ntops", harness.torch, recorder=harness.recorder)
    kinds = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    )
    weight = _Tensor(np.ones((3, 5), dtype=np.float32))
    modules = [("model." + kind, _Linear(weight)) for kind in kinds]
    model = SimpleNamespace(named_modules=lambda: iter(modules))
    input = _Tensor(np.ones((1, 19, 5), dtype=np.float32), weight.device)

    with pytest.raises(RuntimeError, match="synthetic interrupt"):
        with adapter.patched(model):
            for _, module in modules:
                module(input)
            raise RuntimeError("synthetic interrupt")

    assert harness.recorder.calls == {"linear:" + kind: 1 for kind in kinds}
    assert adapter.calls == {kind: 1 for kind in kinds}
    assert all("forward" not in module.__dict__ for _, module in modules)
    assert not adapter._patched
    metadata = adapter.provenance()
    assert len(metadata["patched_modules"]) == len(kinds)
    assert metadata["native_matmul_fallback"] is False
    assert "FP16" in metadata["ntops_numeric_contract"]


def test_bias_is_rejected_before_any_dispatch_or_partial_model_patch(harness):
    adapter = e2e_linear.Adapter("ntops", harness.torch)
    weight = _Tensor(np.ones((3, 5), dtype=np.float32))
    input = _Tensor(np.ones((1, 5), dtype=np.float32), weight.device)
    bias = _Tensor(np.zeros(3, dtype=np.float32), weight.device)

    with pytest.raises(ValueError, match="bias=None"):
        adapter.linear(input, weight, bias)

    first, second = _Linear(weight), _Linear(weight, bias)
    model = SimpleNamespace(
        named_modules=lambda: iter((("first", first), ("second", second)))
    )

    with pytest.raises(ValueError, match="bias-free"):
        with adapter.patched(model):
            raise AssertionError("Invalid model must not be patched.")

    assert "forward" not in first.__dict__
    assert not harness.calls


def test_mismatched_dtype_or_native_device_is_rejected(harness):
    adapter = e2e_linear.Adapter("ntops", harness.torch)
    weight = _Tensor(np.ones((3, 5), dtype=np.float32))
    input = _Tensor(np.ones((1, 5), dtype=np.float16), weight.device)

    with pytest.raises(ValueError, match="share an NPU device"):
        adapter.linear(input, weight)

    input = _Tensor(np.ones((1, 5), dtype=np.float32), SimpleNamespace(type="cpu"))

    with pytest.raises(ValueError, match="share an NPU device"):
        adapter.linear(input, weight)

    assert not harness.calls


def test_counts_are_limited_to_the_recorders_collection_window(harness):
    adapter = e2e_linear.Adapter("ntops", harness.torch, recorder=harness.recorder)
    weight = _Tensor(np.ones((3, 5), dtype=np.float32))
    input = _Tensor(np.ones((1, 5), dtype=np.float32), weight.device)
    harness.recorder.collect = False
    adapter.linear(input, weight, module_name="lm_head")
    harness.recorder.collect = True
    adapter.linear(input, weight, module_name="lm_head")

    assert len(harness.calls) == 2
    assert harness.recorder.calls == {"linear:lm_head": 1}
    assert adapter.calls == {"lm_head": 1}


def test_custom_instance_forward_is_not_silently_replaced(harness):
    adapter = e2e_linear.Adapter("ntops", harness.torch)
    weight = _Tensor(np.ones((3, 5), dtype=np.float32))
    module = _Linear(weight)

    def override(input):
        return input

    module.forward = override
    model = SimpleNamespace(named_modules=lambda: iter((("q_proj", module),)))

    with pytest.raises(ValueError, match="Custom Linear instance"):
        with adapter.patched(model):
            raise AssertionError("Custom semantics must not be discarded.")

    assert module.forward is override
    assert not harness.calls
