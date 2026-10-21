import collections
import contextlib
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.ascend_report import e2e_linear

BM, BN, BK = "dynamic_m", "dynamic_n", "dynamic_k"


def _arrangement(left, right, output):
    return left.tile((BM, BK)), right.tile((BK, BN)), output.tile((BM, BN))


def _application(left, right, output):
    output = left + right  # noqa: F841


class _SymbolicTensor:
    def __init__(self, rank):
        self.rank = rank

    def tile(self, shape):
        return self.rank, shape


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
    state = SimpleNamespace(calls=[], disabled=0, device_contexts=0, fixed_builds=[])

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

    def make(arrangement, application, tensors, **configuration):
        state.fixed_builds.append(
            {
                "arrangement": arrangement,
                "application": application,
                "tiles": arrangement(*tensors),
                "configuration": configuration,
            }
        )
        return generated

    module = SimpleNamespace(
        __name__="ntops_lab.kernels.linear.mm",
        __file__="synthetic/ntops/mm.py",
        kernel=generated,
        run=fail_public_run,
        arrangement=_arrangement,
        application=_application,
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
        "ninetoothed": SimpleNamespace(Tensor=_SymbolicTensor, make=make),
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


def test_fixed_tuning_binds_only_arrangement_meta_parameters(harness):
    adapter = e2e_linear.Adapter("ntops", harness.torch, tuning="fixed")
    assert len(harness.fixed_builds) == 1
    built = harness.fixed_builds[0]
    assert built["arrangement"].__code__ is _arrangement.__code__
    assert built["arrangement"].__globals__ is not _arrangement.__globals__
    assert built["application"] is _application
    assert built["tiles"] == ((2, (32, 32)), (2, (32, 32)), (2, (32, 32)))
    assert (BM, BN, BK) == ("dynamic_m", "dynamic_n", "dynamic_k")
    assert built["configuration"] == {
        "kernel_name": "ntops_lab_mm_fixed_32_32_32",
        "num_warps": 8,
        "num_stages": 1,
    }
    metadata = adapter.provenance()["tuning"]
    assert metadata["autotune_enabled"] is False
    assert metadata["kernel_name"] == "ntops_lab_mm_fixed_32_32_32_npu"


def test_fixed_tuning_retains_baseline_kernel_and_tiles(harness):
    adapter = e2e_linear.Adapter("triton", harness.torch, tuning="fixed")
    metadata = adapter.provenance()["tuning"]
    assert not harness.fixed_builds
    assert metadata["block_sizes"] == [32, 32, 32]
    assert metadata["kernel_name"] == "mm_kernel"
    assert metadata["autotune_enabled"] is False


@pytest.mark.parametrize(
    "arguments",
    ({"tuning": "unknown"}, {"block_sizes": (32, 32)}, {"block_sizes": (32, 24, 32)}),
)
def test_invalid_tuning_is_rejected_before_kernel_import(harness, arguments):
    with pytest.raises(ValueError):
        e2e_linear.Adapter("ntops", harness.torch, **arguments)
    assert not harness.fixed_builds


@pytest.fixture
def packing_harness(harness, monkeypatch):
    state = harness
    state.parameter_roots = []
    state.copy_calls = 0
    state.fail_copy_at = None
    state.max_allocated = 0

    def root_array(values):
        while isinstance(values.base, np.ndarray):
            values = values.base
        return values

    class Parameter(_Tensor):
        def __init__(self, values, device=None):
            super().__init__(values, device)
            self.grad = None
            self.requires_grad = True
            state.parameter_roots.append(weakref.ref(root_array(self.values)))

        @property
        def data(self):
            return _Tensor(self.values, self.device)

        @data.setter
        def data(self, tensor):
            self.values = tensor.values

    def storage(tensor):
        root = root_array(tensor.values)
        pointer, size = root.__array_interface__["data"][0], root.nbytes
        return SimpleNamespace(data_ptr=lambda: pointer, nbytes=lambda: size)

    def contiguous(tensor):
        if tensor.values.flags.c_contiguous:
            return tensor
        state.copy_calls += 1
        if state.copy_calls == state.fail_copy_at:
            raise RuntimeError("synthetic packing allocation failure")
        array = np.ascontiguousarray(tensor.values)
        state.parameter_roots.append(weakref.ref(array))
        return _Tensor(array, tensor.device)

    def allocated(device):
        roots = {id(ref()): ref() for ref in state.parameter_roots if ref() is not None}
        current = sum(array.nbytes for array in roots.values())
        state.max_allocated = max(state.max_allocated, current)
        return current

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 0.25

    monkeypatch.setattr(_Tensor, "untyped_storage", storage, raising=False)
    monkeypatch.setattr(
        _Tensor,
        "storage_offset",
        lambda tensor: (
            (
                tensor.values.__array_interface__["data"][0]
                - root_array(tensor.values).__array_interface__["data"][0]
            )
            // tensor.values.itemsize
        ),
        raising=False,
    )
    monkeypatch.setattr(
        _Tensor, "numel", lambda tensor: tensor.values.size, raising=False
    )
    monkeypatch.setattr(
        _Tensor, "element_size", lambda tensor: tensor.values.itemsize, raising=False
    )
    monkeypatch.setattr(
        _Tensor,
        "is_contiguous",
        lambda tensor: tensor.values.flags.c_contiguous,
        raising=False,
    )
    monkeypatch.setattr(_Tensor, "contiguous", contiguous)
    state.torch.nn.Parameter = Parameter
    state.torch.no_grad = contextlib.nullcontext
    state.torch.inference_mode = lambda enabled=True: contextlib.nullcontext()
    state.torch.npu.Event = Event
    state.torch.npu.synchronize = lambda device=None: None
    state.torch.npu.memory_allocated = allocated
    state.torch.npu.max_memory_allocated = lambda device: state.max_allocated
    state.torch.npu.memory_reserved = allocated
    state.Parameter = Parameter
    state.root_array = root_array

    def model(modules, extra_parameters=(), buffers=()):
        return SimpleNamespace(
            named_modules=lambda: iter(modules),
            named_parameters=lambda remove_duplicate=False: iter(
                [(name + ".weight", module.weight) for name, module in modules]
                + list(extra_parameters)
            ),
            named_buffers=lambda remove_duplicate=False: iter(buffers),
        )

    state.model = model
    return state


@pytest.mark.parametrize("backend", ("ntops", "triton"))
def test_packing_preserves_parameter_identity_values_and_restores_layout(
    packing_harness, backend
):
    state = packing_harness
    weight = state.Parameter(np.arange(15, dtype=np.float32).reshape(3, 5))
    expected = weight.values.copy()
    original_storage = weakref.ref(state.root_array(weight.values))
    module = _Linear(weight)
    adapter = e2e_linear.Adapter(backend, state.torch, weight_layout="packed")
    with adapter.patched(state.model([("q_proj", module)])):
        assert module.weight is weight
        assert weight.stride() == (1, 3)
        assert weight.transpose(0, 1).is_contiguous()
        np.testing.assert_array_equal(weight.values, expected)
        assert original_storage() is None
    assert module.weight is weight
    assert weight.stride() == (5, 1)
    np.testing.assert_array_equal(weight.values, expected)
    statistics_ = adapter.provenance()["weight_preparation"]
    assert statistics_["old_tensor_or_storage_snapshots_retained"] is False
    assert (
        statistics_["packing"]["completed"] and statistics_["restoration"]["completed"]
    )
    assert statistics_["packing"]["copied_bytes"] == expected.nbytes
    assert (
        statistics_["packing"]["maximum_single_matrix_temporary_bytes"]
        == expected.nbytes
    )
    assert (
        statistics_["packing"]["memory_before"]["memory_allocated"]
        == statistics_["packing"]["memory_after"]["memory_allocated"]
    )


def test_shared_parameter_is_packed_once_and_both_modules_stay_consistent(
    packing_harness,
):
    state = packing_harness
    weight = state.Parameter(np.arange(15, dtype=np.float32).reshape(3, 5))
    modules = [("q_proj", _Linear(weight)), ("k_proj", _Linear(weight))]
    adapter = e2e_linear.Adapter("ntops", state.torch, weight_layout="packed")
    with adapter.patched(state.model(modules)):
        assert modules[0][1].weight is modules[1][1].weight is weight
        assert state.copy_calls == 1
        assert weight.stride() == (1, 3)
    assert state.copy_calls == 2
    stats = adapter.provenance()["weight_preparation"]
    assert stats["unique_parameter_count"] == 1
    assert stats["selected_module_count"] == 2
    assert stats["shared_parameter_module_count"] == 1
    assert weight.stride() == (5, 1)


@pytest.mark.parametrize(
    "alias_kind",
    ("different_parameter", "nonlinear_parameter", "buffer", "noncontiguous"),
)
def test_unsafe_weight_aliases_or_layouts_are_rejected_before_copy(
    packing_harness, alias_kind
):
    state = packing_harness
    weight = state.Parameter(np.arange(15, dtype=np.float32).reshape(3, 5))
    modules = [("q_proj", _Linear(weight))]
    extra, buffers = [], []
    if alias_kind == "different_parameter":
        modules.append(("k_proj", _Linear(state.Parameter(weight.values))))
    elif alias_kind == "nonlinear_parameter":
        extra.append(("embed_tokens.weight", weight))
    elif alias_kind == "buffer":
        buffers.append(("weight_cache", _Tensor(weight.values, weight.device)))
    else:
        weight.values = weight.values.T
    adapter = e2e_linear.Adapter("ntops", state.torch, weight_layout="packed")
    with pytest.raises(ValueError):
        with adapter.patched(state.model(modules, extra, buffers)):
            raise AssertionError("Unsafe weight layout must not be entered.")
    assert state.copy_calls == 0
    assert not adapter._patched
    assert "forward" not in modules[0][1].__dict__


def test_partial_packing_failure_restores_already_packed_parameters(packing_harness):
    state = packing_harness
    weights = [
        state.Parameter(np.arange(15, dtype=np.float32).reshape(3, 5)) for _ in range(2)
    ]
    modules = [("q_proj", _Linear(weights[0])), ("k_proj", _Linear(weights[1]))]
    state.fail_copy_at = 2
    adapter = e2e_linear.Adapter("ntops", state.torch, weight_layout="packed")
    with pytest.raises(RuntimeError, match="synthetic packing allocation"):
        with adapter.patched(state.model(modules)):
            raise AssertionError("The second packing operation must fail.")
    assert all(weight.stride() == (5, 1) for weight in weights)
    assert all("forward" not in module.__dict__ for _, module in modules)
    assert not adapter._patched
    stats = adapter.provenance()["weight_preparation"]
    assert stats["packing"]["completed"] is False
    assert stats["restoration"]["completed"] is True
    assert stats["restoration"]["parameters_processed"] == 1


def test_exception_inside_model_context_restores_all_packed_weights(packing_harness):
    state = packing_harness
    weights = [
        state.Parameter(np.full((3, 5), number, dtype=np.float32)) for number in (1, 2)
    ]
    modules = [("q_proj", _Linear(weights[0])), ("k_proj", _Linear(weights[1]))]
    adapter = e2e_linear.Adapter("ntops", state.torch, weight_layout="packed")
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        with adapter.patched(state.model(modules)):
            raise RuntimeError("synthetic inference failure")
    for number, weight in enumerate(weights, 1):
        assert weight.stride() == (5, 1)
        assert np.all(weight.values == number)
    assert (
        adapter.provenance()["weight_preparation"]["restoration"]["completed"] is True
    )
