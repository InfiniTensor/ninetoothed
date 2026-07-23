import functools
import gc
import inspect
import uuid
import weakref
from types import SimpleNamespace

import pytest
import torch

import ninetoothed
import ninetoothed.auto_tuner as auto_tuner
import ninetoothed.backends.materializers.triton as triton_materializer
import ninetoothed.compiler.runtime as runtime
from ninetoothed import Tensor
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.ir import LaunchABI, LaunchBinding
from tests.utils import get_available_devices

_prepared_test_kernel = None


def _arrangement(input, other, output):
    return tuple(tensor.tile((64,)) for tensor in (input, other, output))


def _application(input, other, output):
    output = input + other  # noqa: F841


def _control_dependent_inplace_arrangement(output):
    return (output.tile((1,)),)


def _loop_carried_arrangement(input, output):
    return (input.tile((1,)), output.tile((1,)))


def _control_dependent_inplace_application(output):
    if output[0] > 1:
        output[0] = 0
    elif output[0] > 0:
        output[0] = 2
    else:
        output[0] = 1


def _loop_carried_application(input, output):
    value = input

    for _ in range(2):
        output = value  # noqa: F841
        value = value + 1


class _FakeTensor:
    def __init__(
        self,
        shape,
        *,
        stride=None,
        dtype="float32",
        device="cuda:0",
        data_ptr=None,
    ):
        self.shape = tuple(shape)
        self._stride = tuple(stride or _contiguous_stride(shape))
        self.dtype = dtype
        self.device = device
        self._data_ptr = id(self) if data_ptr is None else data_ptr
        self._storage_offset = 0

    def stride(self):
        return self._stride

    def data_ptr(self):
        return self._data_ptr

    def element_size(self):
        return 4

    def storage_offset(self):
        return self._storage_offset

    def numel(self):
        result = 1

        for size in self.shape:
            result *= size
        return result


def _contiguous_stride(shape):
    result = []
    stride = 1

    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return tuple(reversed(result))


class _FakeTuner:
    def __init__(self, candidates, key_function, *, selected_index=None):
        self._funcs = tuple(candidates)
        self._best_func = {}
        self._key_function = key_function
        self._selected_index = selected_index
        self.calls = 0

    def _make_arg_key(self, args, kwargs):
        return self._key_function(args, kwargs)

    def __call__(self, *args, **kwargs):
        self.calls += 1

        for candidate in self._funcs:
            candidate(*args, **kwargs)

        key = self._make_arg_key(args, kwargs)
        selected = (
            self._funcs[self._selected_index]
            if self._selected_index is not None
            else self._funcs[0]
            if args[0].shape[0] == 4
            else self._funcs[1]
        )
        self._best_func[key] = selected

        return selected(*args, **kwargs)


def _runtime_fixture(*, with_constexpr=False):
    public_args = ("value", "scale") if with_constexpr else ("value",)
    bindings = [LaunchBinding(name="value", kind="tensor", source="value")]

    if with_constexpr:
        bindings.append(LaunchBinding(name="scale", kind="constexpr", source="scale"))

    abi = LaunchABI(public_args=public_args, kernel_args=tuple(bindings))
    calls = []

    def candidate(name):
        return runtime._runtime_wrapper(
            lambda *values: calls.append((name, values)) or values[0],
            abi,
        )

    candidates = (candidate("first"), candidate("second"))
    compilation = SimpleNamespace(
        launch_abi=abi,
        kernel=SimpleNamespace(tensors=()),
    )
    tuner = _FakeTuner(
        candidates,
        lambda args, kwargs: triton_materializer._triton_specialization_key(
            compilation, args, kwargs
        ),
    )
    handle = SimpleNamespace(_selected_tuning_candidate=None)
    launch = triton_materializer._tuned_runtime_launch(
        tuner,
        dict(zip(candidates, ({"id": "first"}, {"id": "second"}))),
        handle,
        compilation,
    )

    return launch, tuner, handle, calls


@pytest.mark.parametrize(
    "candidate",
    (
        None,
        {
            "id": "single",
            "num_warps": 4,
            "num_stages": 1,
            "meta_parameters": {},
        },
    ),
    ids=("plain", "single-candidate"),
)
def test_triton_single_and_plain_materialization_use_verified_launch(
    candidate, tmp_path, monkeypatch
):
    abi = LaunchABI(
        public_args=("value",),
        kernel_args=(LaunchBinding(name="value", kind="tensor", source="value"),),
    )
    raw_calls = []

    def raw_launch(*values, **kwargs):
        raw_calls.append((values, kwargs))

        return values[0]

    class FakeHandle:
        def __init__(self, compilation, kernel, launch, source):
            del compilation, kernel, source
            self._launch = launch

        def __call__(self, *args, **kwargs):
            return self._launch(*args, **kwargs)

    compilation = SimpleNamespace(
        artifact=SimpleNamespace(
            kernel_name="verified_runtime",
            primary_source="",
            entrypoint="launch",
        ),
        launch_abi=abi,
        launch_plan=SimpleNamespace(
            tuning_candidates=() if candidate is None else (candidate,)
        ),
        kernel=SimpleNamespace(tensors=()),
    )
    public_calls = 0
    original_public = runtime._public_values

    def public(*args, **kwargs):
        nonlocal public_calls
        public_calls += 1

        return original_public(*args, **kwargs)

    monkeypatch.setattr(runtime, "Handle", FakeHandle)
    monkeypatch.setattr(runtime, "_public_values", public)
    monkeypatch.setattr(triton_materializer, "_ensure_c_compiler", lambda: None)
    monkeypatch.setattr(
        triton_materializer,
        "compilation_cache_key",
        lambda compilation: "verified-runtime",
    )
    monkeypatch.setattr(
        triton_materializer,
        "write_source",
        lambda *args, **kwargs: tmp_path / "verified_runtime.py",
    )
    monkeypatch.setattr(
        runtime,
        "import_python_module",
        lambda source: SimpleNamespace(launch=raw_launch),
    )
    monkeypatch.setattr(triton_materializer, "TRITON_CACHE_DIR", tmp_path / "cache")
    handle = triton_materializer._materialize(compilation)
    value = _FakeTensor((4,))

    assert handle(value) is value
    assert handle(value) is value
    assert len(raw_calls) == 2
    assert public_calls == 1
    assert len(inspect.getclosurevars(handle._launch).nonlocals["prepared_calls"]) == 1


def _verified_runtime_fixture(*, with_constexpr=False, outputs=()):
    public_args = ("value", "scale") if with_constexpr else ("value",)
    bindings = [LaunchBinding(name="value", kind="tensor", source="value")]

    if with_constexpr:
        bindings.append(LaunchBinding(name="scale", kind="constexpr", source="scale"))

    abi = LaunchABI(
        public_args=public_args,
        kernel_args=tuple(bindings),
        outputs=outputs,
    )
    calls = []
    wrapped = runtime._runtime_wrapper(
        lambda *values: calls.append(values),
        abi,
    )

    return runtime._verified_runtime_launch(wrapped), calls


def test_verified_runtime_launch_rebinds_zero_dimensional_value(monkeypatch):
    binding_calls = 0
    original_bound = runtime._bound_values

    def bound(*args, **kwargs):
        nonlocal binding_calls
        binding_calls += 1

        return original_bound(*args, **kwargs)

    monkeypatch.setattr(runtime, "_bound_values", bound)
    launch, calls = _verified_runtime_fixture(with_constexpr=True)
    value = _FakeTensor((4,))
    scale = torch.tensor(2)

    launch(value, scale)
    launch(value, scale)
    scale.fill_(3)
    launch(value, scale)

    assert [values[-1] for values in calls] == [2, 2, 3]
    assert binding_calls == 2


def test_verified_runtime_launch_keeps_eight_recent_identities(monkeypatch):
    binding_calls = 0
    original_bound = runtime._bound_values

    def bound(*args, **kwargs):
        nonlocal binding_calls
        binding_calls += 1

        return original_bound(*args, **kwargs)

    monkeypatch.setattr(runtime, "_bound_values", bound)
    launch, _ = _verified_runtime_fixture()
    values = [_FakeTensor((size,)) for size in range(1, 18)]

    for value in values:
        launch(value)

    nonlocals = inspect.getclosurevars(launch).nonlocals
    assert len(nonlocals["prepared_calls"]) == 8
    assert binding_calls == 17

    launch(values[0])
    assert binding_calls == 18


def test_verified_runtime_launch_handles_zero_without_calling_kernel():
    launch, calls = _verified_runtime_fixture()
    nonempty = _FakeTensor((4,))
    empty = _FakeTensor((0,))

    assert launch(nonempty) is None
    assert launch(empty) is None
    assert launch(nonempty) is None
    assert calls == [(nonempty,), (nonempty,)]


def test_verified_runtime_launch_preserves_argument_errors():
    launch, _ = _verified_runtime_fixture(with_constexpr=True)
    value = _FakeTensor((4,))

    launch(value, 2)

    with pytest.raises(TypeError, match="passed twice"):
        launch(value, 2, scale=2)

    with pytest.raises(TypeError, match="Missing kernel arguments"):
        launch(value)

    with pytest.raises(TypeError, match="Unknown kernel arguments"):
        launch(value, 2, unknown=True)


def test_verified_runtime_fast_key_defers_invalid_tensor_to_validation():
    abi = LaunchABI(
        public_args=("first", "second", "scale"),
        kernel_args=(
            LaunchBinding(name="first", kind="tensor", source="first"),
            LaunchBinding(name="second", kind="tensor", source="second"),
            LaunchBinding(name="scale", kind="scalar", source="scale"),
        ),
    )
    specs = tuple(
        SimpleNamespace(
            name=name,
            ndim=1,
            dtype=None,
            attrs={"source_ndim": 1},
        )
        for name in ("first", "second")
    )
    wrapped = runtime._runtime_wrapper(lambda *_values: None, abi, specs=specs)
    launch = runtime._verified_runtime_launch(wrapped)
    first = _FakeTensor((4,))
    second = _FakeTensor((4,))

    launch(first, second, scale=1)

    with pytest.raises(TypeError, match="`first` must be a tensor"):
        launch(None, second, scale=1)


def test_verified_runtime_four_buffer_call_has_no_cached_output_field():
    names = ("first", "second", "third", "output")
    abi = LaunchABI(
        public_args=names,
        kernel_args=tuple(
            LaunchBinding(name=name, kind="tensor", source=name) for name in names
        ),
        outputs=("output",),
    )
    wrapped = runtime._runtime_wrapper(lambda *values: None, abi)
    launch = runtime._verified_runtime_launch(wrapped)
    buffers = tuple(_FakeTensor((4,)) for _ in names)

    assert launch(*buffers) is buffers[-1]
    assert launch(*buffers) is buffers[-1]
    prepared = inspect.getclosurevars(launch).nonlocals["active"]
    assert not hasattr(prepared, "output")
    assert not hasattr(prepared, "result")


def test_verified_runtime_cache_does_not_retain_tensor_or_flattened_view():
    abi = LaunchABI(
        public_args=("value",),
        kernel_args=(LaunchBinding(name="value", kind="tensor", source="value"),),
    )
    view_refs = []
    invoked_refs = []

    def prepare_invocation(values, _static_values, _call_sources):
        view_refs.append(weakref.ref(values[0]))

        def invoke(current_values, _args, _kwargs):
            invoked_refs.append(weakref.ref(current_values[0]))

        return invoke

    wrapped = runtime._runtime_wrapper(
        lambda _value: None,
        abi,
        prepare_invocation=prepare_invocation,
    )
    launch = runtime._verified_runtime_launch(wrapped)
    value = torch.arange(8)
    value_ref = weakref.ref(value)

    launch(value)
    launch(value)
    gc.collect()

    nonlocals = inspect.getclosurevars(launch).nonlocals
    assert len(nonlocals["prepared_calls"]) == 1
    assert len(view_refs) == 1
    assert all(reference() is None for reference in view_refs)
    assert len(invoked_refs) == 2
    assert all(reference() is None for reference in invoked_refs)
    assert torch.equal(value, torch.arange(8))

    del value
    gc.collect()

    assert value_ref() is None
    assert nonlocals["prepared_calls"] == {}
    assert inspect.getclosurevars(launch).nonlocals["active"] is None


def test_verified_runtime_uses_prepared_low_level_invocation():
    abi = LaunchABI(
        public_args=("value",),
        kernel_args=(LaunchBinding(name="value", kind="tensor", source="value"),),
    )
    prepared_values = []
    invoked_values = []
    raw_calls = []

    def prepare_invocation(values, _static_values, _call_sources):
        prepared_values.append(values)

        def invoke(current_values, _args, _kwargs):
            invoked_values.append(current_values)

        return invoke

    wrapped = runtime._runtime_wrapper(
        lambda *values: raw_calls.append(values),
        abi,
        prepare_invocation=prepare_invocation,
    )
    launch = runtime._verified_runtime_launch(wrapped)
    value = _FakeTensor((4,))

    assert launch(value) is None
    assert launch(value) is None
    assert prepared_values == [(value,)]
    assert invoked_values == [(value,), (value,)]
    assert raw_calls == []


def test_triton_prepared_invocation_caches_grid_without_launching(monkeypatch):
    bound_grids = []
    kernel_calls = []

    class FakeKernel:
        def __getitem__(self, grid):
            bound_grids.append(grid)

            def launch(*args, **kwargs):
                kernel_calls.append((args, kwargs))

                return object()

            return launch

    kernel = FakeKernel()
    monkeypatch.setitem(globals(), "_prepared_test_kernel", kernel)

    def generated_launch(value, size, _ninetoothed_num_warps=4):
        if size <= 0:
            raise ValueError("Size must be positive.")

        _prepared_test_kernel[(size,)](
            value,
            size,
            BLOCK=1,
            num_warps=_ninetoothed_num_warps,
        )

        return value

    function = functools.partial(generated_launch, _ninetoothed_num_warps=8)
    prepare = triton_materializer._triton_prepare_invocation(function, kernel)

    assert prepare is not None
    assert bound_grids == []
    invocation = prepare(("pointer", 4))
    assert bound_grids == [(4,)]
    assert kernel_calls == []

    assert invocation(("pointer", 4), (), {}) is None
    assert invocation(("pointer", 4), (), {}) is None
    assert kernel_calls == [
        (("pointer", 4), {"BLOCK": 1, "num_warps": 8}),
        (("pointer", 4), {"BLOCK": 1, "num_warps": 8}),
    ]

    value = _FakeTensor((4,))
    value_ref = weakref.ref(value)
    unowned_invocation = prepare(
        (value, 4),
        (None, 4),
        (("positional", 0), None),
    )
    assert unowned_invocation((), (value,), {}) is None
    kernel_calls.clear()
    del value
    gc.collect()

    assert unowned_invocation is not None
    assert not unowned_invocation.requires_values
    assert value_ref() is None

    with pytest.raises(ValueError, match="positive"):
        prepare(("pointer", 0))


def test_triton_direct_winner_reuses_verified_binding_and_restores_aba(monkeypatch):
    public_calls = 0
    binding_calls = 0
    original_public = runtime._public_values
    original_bound = runtime._bound_values

    def public(*args, **kwargs):
        nonlocal public_calls
        public_calls += 1

        return original_public(*args, **kwargs)

    def bound(*args, **kwargs):
        nonlocal binding_calls
        binding_calls += 1

        return original_bound(*args, **kwargs)

    monkeypatch.setattr(runtime, "_public_values", public)
    monkeypatch.setattr(runtime, "_bound_values", bound)
    launch, tuner, handle, calls = _runtime_fixture()
    first = _FakeTensor((4,))
    second = _FakeTensor((8,), stride=(2,))

    assert launch(first) is first
    assert handle._selected_tuning_candidate == {"id": "first"}
    assert launch(second) is second
    assert handle._selected_tuning_candidate == {"id": "second"}
    assert tuner.calls == 2
    public_calls = binding_calls = 0
    calls.clear()

    assert launch(first) is first
    assert handle._selected_tuning_candidate == {"id": "first"}
    assert calls == [("first", (first,))]
    assert tuner.calls == 2
    assert public_calls == 0
    assert binding_calls == 0


def test_triton_alias_selection_does_not_pollute_non_alias_tuning():
    abi = LaunchABI(
        public_args=("value", "output"),
        kernel_args=(
            LaunchBinding(name="value", kind="tensor", source="value"),
            LaunchBinding(
                name="output", kind="tensor", source="output", access="write"
            ),
        ),
        outputs=("output",),
    )
    calls = []

    def candidate(name):
        return runtime._runtime_wrapper(
            lambda *values: calls.append((name, values)) or object(),
            abi,
        )

    candidates = (candidate("first"), candidate("second"))
    tuner = _FakeTuner(
        candidates,
        lambda args, kwargs: "shared-key",
        selected_index=1,
    )
    handle = SimpleNamespace(_selected_tuning_candidate=None)
    launch = triton_materializer._tuned_runtime_launch(
        tuner,
        dict(zip(candidates, ({"id": "first"}, {"id": "second"}))),
        handle,
        SimpleNamespace(launch_abi=abi, kernel=SimpleNamespace(tensors=())),
    )
    first_value = _FakeTensor((4,))
    first_output = _FakeTensor((4,))

    assert launch(first_value, output=first_output) is first_output
    assert tuner.calls == 1
    assert [name for name, _ in calls] == ["first", "second", "second"]
    assert tuner._best_func["shared-key"] is candidates[1]
    assert handle._selected_tuning_candidate == {"id": "second"}

    calls.clear()
    aliased = _FakeTensor((4,))

    assert launch(aliased, output=aliased) is aliased
    assert tuner.calls == 1
    assert [name for name, _ in calls] == ["first"]
    assert handle._selected_tuning_candidate == {"id": "first"}
    assert tuner._best_func["shared-key"] is candidates[1]

    calls.clear()
    value = _FakeTensor((4,))
    output = _FakeTensor((4,))

    assert launch(value, output=output) is output
    assert tuner.calls == 1
    assert [name for name, _ in calls] == ["second"]
    assert handle._selected_tuning_candidate == {"id": "second"}


@pytest.mark.parametrize("kind", ("scalar", "constexpr"))
def test_triton_tensor_scalar_alias_uses_alias_safe_selection(kind):
    abi = LaunchABI(
        public_args=("scale", "output"),
        kernel_args=(
            LaunchBinding(name="scale", kind=kind, source="scale"),
            LaunchBinding(
                name="output",
                kind="tensor",
                source="output",
                access="write",
            ),
        ),
        outputs=("output",),
    )
    calls = []

    def candidate(name):
        return runtime._runtime_wrapper(
            lambda *_values: calls.append(name),
            abi,
        )

    candidates = (candidate("first"), candidate("second"))
    tuner = _FakeTuner(candidates, lambda args, kwargs: "shared-key")
    handle = SimpleNamespace(_selected_tuning_candidate=None)
    launch = triton_materializer._tuned_runtime_launch(
        tuner,
        dict(zip(candidates, ({"id": "first"}, {"id": "second"}))),
        handle,
        SimpleNamespace(launch_abi=abi, kernel=SimpleNamespace(tensors=())),
    )
    output = torch.zeros(4)

    assert launch(output[0], output) is output
    assert calls == ["first"]
    assert tuner.calls == 0


def test_triton_alias_signature_uses_absolute_spans_and_fails_closed():
    abi = LaunchABI(
        public_args=("value", "output"),
        kernel_args=(
            LaunchBinding(name="value", kind="tensor", source="value", access="read"),
            LaunchBinding(
                name="output",
                kind="tensor",
                source="output",
                access="write",
            ),
        ),
        outputs=("output",),
    )
    output = _FakeTensor((4,), data_ptr=1024)

    overlapping = _FakeTensor((4,), data_ptr=1032)
    assert triton_materializer._runtime_alias_signature(
        abi, {"value": overlapping, "output": output}
    )

    unknown = SimpleNamespace(
        shape=(4,),
        stride=lambda: (1,),
        data_ptr=lambda: 2048,
        device="cuda:0",
    )
    assert triton_materializer._runtime_alias_signature(
        abi, {"value": unknown, "output": output}
    )

    legacy_output_abi = LaunchABI(
        public_args=("output",),
        kernel_args=(LaunchBinding(name="output", kind="tensor", source="output"),),
        outputs=("output",),
    )
    assert triton_materializer._runtime_alias_signature(
        legacy_output_abi, {"output": output}
    )

    mixed_access_abi = LaunchABI(
        public_args=("output",),
        kernel_args=(
            LaunchBinding(name="read", kind="tensor", source="output", access="read"),
            LaunchBinding(name="write", kind="tensor", source="output", access="write"),
        ),
        outputs=("output",),
    )
    assert triton_materializer._runtime_alias_signature(
        mixed_access_abi, {"output": output}
    )


def test_triton_read_write_binding_uses_alias_safe_selection():
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_control_dependent_inplace_arrangement,
            application=_control_dependent_inplace_application,
            tensors=(Tensor(1),),
            backend="triton",
            num_warps=(4, 8),
        )
    )
    abi = compilation.launch_abi
    binding = next(binding for binding in abi.kernel_args if binding.kind == "tensor")
    calls = []

    def candidate(name):
        def increment(value, *_metadata):
            calls.append(name)

            return value.add_(1)

        return runtime._runtime_wrapper(increment, abi)

    candidates = (candidate("first"), candidate("second"))
    tuner = _FakeTuner(candidates, lambda args, kwargs: "shared-key")
    handle = SimpleNamespace(_selected_tuning_candidate=None)
    launch = triton_materializer._tuned_runtime_launch(
        tuner,
        dict(zip(candidates, ({"id": "first"}, {"id": "second"}))),
        handle,
        SimpleNamespace(launch_abi=abi, kernel=SimpleNamespace(tensors=())),
    )
    output = torch.zeros(4)

    assert binding.access == "read_write"
    assert launch(output) is output
    assert torch.equal(output, torch.ones(4))
    assert calls == ["first"]
    assert tuner.calls == 0


def test_triton_loop_carried_binding_access_preserves_input_dependency():
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_loop_carried_arrangement,
            application=_loop_carried_application,
            tensors=(Tensor(1), Tensor(1)),
            backend="triton",
        )
    )
    access = {
        binding.source: binding.access
        for binding in compilation.launch_abi.kernel_args
        if binding.kind == "tensor"
    }

    assert access == {"input": "read", "output": "write"}


def test_triton_alias_selection_uses_jagged_binding_access():
    abi = LaunchABI(
        public_args=("value", "output"),
        kernel_args=(
            LaunchBinding(
                name="value_values",
                kind="jagged_values",
                source="value",
                access="read",
            ),
            LaunchBinding(
                name="value_offsets",
                kind="jagged_offsets",
                source="value",
                access="read",
            ),
            LaunchBinding(
                name="output_values",
                kind="jagged_values",
                source="output",
                access="write",
            ),
            LaunchBinding(
                name="output_offsets",
                kind="jagged_offsets",
                source="output",
                access="read",
            ),
        ),
        outputs=("output",),
    )
    calls = []

    def candidate(name):
        return runtime._runtime_wrapper(
            lambda *values: calls.append((name, values)),
            abi,
        )

    candidates = (candidate("first"), candidate("second"))
    tuner = _FakeTuner(candidates, lambda args, kwargs: "shared-key")
    handle = SimpleNamespace(_selected_tuning_candidate=None)
    launch = triton_materializer._tuned_runtime_launch(
        tuner,
        dict(zip(candidates, ({"id": "first"}, {"id": "second"}))),
        handle,
        SimpleNamespace(launch_abi=abi, kernel=SimpleNamespace(tensors=())),
    )
    values = torch.arange(8.0)
    value = torch.nested.nested_tensor_from_jagged(values, torch.tensor((0, 4, 8)))
    output = torch.nested.nested_tensor_from_jagged(values, torch.tensor((0, 2, 8)))

    assert launch(value, output=output) is output
    assert tuner.calls == 0
    assert [name for name, _ in calls] == ["first"]
    assert handle._selected_tuning_candidate == {"id": "first"}

    calls.clear()
    offsets = torch.tensor((0, 4, 8))
    value = torch.nested.nested_tensor_from_jagged(torch.arange(8.0), offsets)
    output = torch.nested.nested_tensor_from_jagged(torch.empty(8), offsets)

    assert launch(value, output=output) is output
    assert tuner.calls == 1
    assert [name for name, _ in calls] == ["first", "second", "second"]
    assert handle._selected_tuning_candidate == {"id": "second"}


def test_triton_alias_selection_handles_candidate_failures_safely():
    abi = LaunchABI(
        public_args=("value", "output"),
        kernel_args=(
            LaunchBinding(name="value", kind="tensor", source="value"),
            LaunchBinding(
                name="output", kind="tensor", source="output", access="write"
            ),
        ),
        outputs=("output",),
    )
    calls = []

    def candidate(name, *, fails=False):
        def invoke(*_values):
            calls.append(name)

            if fails:
                raise RuntimeError("Candidate unavailable.")

        return runtime._runtime_wrapper(invoke, abi)

    candidates = (candidate("first", fails=True), candidate("second"))
    tuner = _FakeTuner(candidates, lambda args, kwargs: "shared-key")
    handle = SimpleNamespace(_selected_tuning_candidate=None)
    launch = triton_materializer._tuned_runtime_launch(
        tuner,
        dict(zip(candidates, ({"id": "first"}, {"id": "second"}))),
        handle,
        SimpleNamespace(launch_abi=abi, kernel=SimpleNamespace(tensors=())),
    )
    value = _FakeTensor((4,))

    with pytest.raises(RuntimeError, match="Candidate unavailable"):
        launch(value, output=value)

    assert calls == ["first"]
    assert tuner.calls == 0
    assert tuner._best_func == {}
    assert handle._selected_tuning_candidate is None

    calls.clear()
    candidates = (candidate("first"), candidate("second"))

    def fail_prepare(*_args, **_kwargs):
        raise RuntimeError("Candidate did not prepare.")

    candidates[0]._ninetoothed_prepare = fail_prepare
    tuner = _FakeTuner(candidates, lambda args, kwargs: "shared-key")
    launch = triton_materializer._tuned_runtime_launch(
        tuner,
        dict(zip(candidates, ({"id": "first"}, {"id": "second"}))),
        handle,
        SimpleNamespace(launch_abi=abi, kernel=SimpleNamespace(tensors=())),
    )

    assert launch(value, output=value) is value
    assert calls == ["first"]
    assert handle._selected_tuning_candidate == {"id": "first"}

    calls.clear()
    candidates = (candidate("first"), candidate("second"))
    candidates[0]._ninetoothed_prepare = fail_prepare
    tuner = _FakeTuner(
        candidates,
        lambda args, kwargs: "shared-key",
        selected_index=0,
    )
    launch = triton_materializer._tuned_runtime_launch(
        tuner,
        dict(zip(candidates, ({"id": "first"}, {"id": "second"}))),
        handle,
        SimpleNamespace(launch_abi=abi, kernel=SimpleNamespace(tensors=())),
    )

    assert launch(_FakeTensor((4,)), output=_FakeTensor((4,))) is not None
    assert calls == ["first", "second", "first"]
    assert handle._selected_tuning_candidate == {"id": "first"}


@pytest.mark.parametrize(
    ("field", "value", "expected_tuner_calls"),
    (
        ("shape", (4, 1), 2),
        ("_stride", (2,), 2),
        ("dtype", "float16", 2),
        ("device", "cuda:1", 2),
        ("_data_ptr", 1, 1),
        ("_storage_offset", 1, 1),
    ),
)
def test_prepared_launch_rebinds_after_tensor_contract_change(
    field, value, expected_tuner_calls
):
    launch, tuner, _, _ = _runtime_fixture()
    tensor = _FakeTensor((4,))

    assert launch(tensor) is tensor
    assert tuner.calls == 1
    setattr(tensor, field, value)
    assert launch(tensor) is tensor
    assert tuner.calls == expected_tuner_calls


def test_triton_direct_winner_validates_constexpr_and_argument_errors():
    launch, tuner, _, _ = _runtime_fixture(with_constexpr=True)
    tensor = _FakeTensor((4,))

    assert launch(tensor, 2) is tensor
    assert launch(tensor, 2) is tensor
    assert tuner.calls == 1
    assert launch(tensor, 3) is tensor
    assert tuner.calls == 2
    assert launch(tensor, 2) is tensor
    assert tuner.calls == 2

    with pytest.raises(TypeError, match="passed twice"):
        launch(tensor, 2, scale=2)

    with pytest.raises(TypeError, match="Missing kernel arguments"):
        launch(tensor)

    with pytest.raises(TypeError, match="Unknown kernel arguments"):
        launch(tensor, 2, unknown=True)


def test_zero_dimensional_constexpr_mutation_rebinds_current_value():
    launch, tuner, _, calls = _runtime_fixture(with_constexpr=True)
    tensor = _FakeTensor((4,))
    scale = torch.tensor(2)

    assert launch(tensor, scale) is tensor
    calls.clear()
    assert launch(tensor, scale) is tensor
    assert calls == [("first", (tensor, 2))]

    scale.fill_(3)
    calls.clear()
    assert launch(tensor, scale) is tensor
    assert [values[-1] for _, values in calls] == [3, 3, 3]
    assert tuner.calls == 2

    abi = LaunchABI(
        public_args=("value", "scale"),
        kernel_args=(
            LaunchBinding(name="value", kind="tensor", source="value"),
            LaunchBinding(name="scale", kind="constexpr", source="scale"),
        ),
    )
    compilation = SimpleNamespace(launch_abi=abi)
    scale.fill_(2)
    first_key = triton_materializer._triton_specialization_key(
        compilation, (tensor, scale), {}
    )
    scale.fill_(3)
    second_key = triton_materializer._triton_specialization_key(
        compilation, (tensor, scale), {}
    )
    assert first_key != second_key


def test_triton_winner_proofs_share_the_bounded_prepared_lru():
    launch, tuner, _, _ = _runtime_fixture()

    for size in range(1, 18):
        tensor = _FakeTensor((size,))
        assert launch(tensor) is tensor

    nonlocals = inspect.getclosurevars(launch).nonlocals
    assert "selected_by_key" not in nonlocals
    assert len(nonlocals["prepared_calls"]) == 8
    assert tuner.calls == 17

    first = _FakeTensor((1,))
    assert launch(first) is first
    assert tuner.calls == 18


def test_prepared_launch_resolves_output_from_the_current_call():
    abi = LaunchABI(
        public_args=("value", "output"),
        kernel_args=(
            LaunchBinding(name="value", kind="tensor", source="value"),
            LaunchBinding(name="output", kind="tensor", source="output"),
        ),
        outputs=("output",),
    )
    internal_result = object()
    wrapped = runtime._runtime_wrapper(
        lambda value, output: internal_result,
        abi,
    )
    value = _FakeTensor((4,))
    output = _FakeTensor((4,))
    prepared = wrapped._ninetoothed_prepare((value, output), {})

    assert wrapped(value, output) is output
    assert not hasattr(prepared, "output")
    assert not hasattr(prepared, "args")
    assert not hasattr(prepared, "kwargs")
    assert prepared.guard.positional[1].identity == id(output)
    assert prepared.guard.positional[1].value is None
    assert wrapped._ninetoothed_invoke_prepared(prepared, (value, output), {}) is output


def test_triton_zero_size_does_not_disturb_nonempty_prepared_call():
    launch, tuner, _, calls = _runtime_fixture()
    nonempty = _FakeTensor((4,))
    empty = _FakeTensor((0,))

    assert launch(nonempty) is nonempty
    assert tuner.calls == 1
    calls.clear()
    assert launch(empty) is None
    assert calls == []
    assert tuner.calls == 1
    assert launch(nonempty) is nonempty
    assert tuner.calls == 1


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


@pytest.mark.parametrize("device", get_available_devices())
def test_triton_prepared_cache_releases_gpu_tensor_storage(device):
    handle = ninetoothed.make(
        _arrangement,
        _application,
        tuple(Tensor(shape=(1 << 20,), dtype=ninetoothed.float32) for _ in range(3)),
        backend="triton",
        kernel_name=f"runtime_weak_cache_{uuid.uuid4().hex}",
    )
    warm_input = torch.randn(1 << 20, device=device)
    warm_other = torch.randn_like(warm_input)
    warm_output = torch.empty_like(warm_input)
    handle(warm_input, warm_other, warm_output)
    torch.cuda.synchronize(device)
    del warm_input, warm_other, warm_output
    gc.collect()
    baseline = torch.cuda.memory_allocated(device)

    input = torch.randn(1 << 20, device=device)
    other = torch.randn_like(input)
    output = torch.empty_like(input)
    references = tuple(weakref.ref(value) for value in (input, other, output))
    handle(input, other, output)
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device)
    del input, other, output
    gc.collect()
    torch.cuda.synchronize(device)

    assert allocated > baseline
    assert all(reference() is None for reference in references)
    assert torch.cuda.memory_allocated(device) <= baseline

    replacement_input = torch.randn(1 << 20, device=device)
    replacement_other = torch.randn_like(replacement_input)
    replacement_output = torch.empty_like(replacement_input)
    handle(replacement_input, replacement_other, replacement_output)
    torch.cuda.synchronize(device)

    assert torch.allclose(replacement_output, replacement_input + replacement_other)


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
