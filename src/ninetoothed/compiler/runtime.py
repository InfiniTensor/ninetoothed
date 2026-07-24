"""Materialize backend artifacts into callable runtime handles."""

import copy
import ctypes
import importlib.util
import os
import shutil
import sys
import tempfile
import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from ninetoothed.backends.core import BuiltArtifact, Target
from ninetoothed.compiler.cache import (
    atomic_write_bytes,
    compilation_cache_key,
    write_manifest,
    write_source,
)
from ninetoothed.ir import LaunchABI, LaunchBinding, ir_to_dict


class KernelLaunchError(RuntimeError):
    """Raised when a materialized backend launcher reports an error."""

    def __init__(self, error_code):
        super().__init__(f"Kernel launch failed with error code: {error_code}.")


def overflow_terms(argument_names, tensor_ndims):
    """Return C launcher guards for shape and stride values outside int32."""
    int32_min = -(2**31)
    int32_max = 2**31 - 1

    return tuple(
        term
        for name, ndim in zip(argument_names, tensor_ndims)
        for dim in range(ndim)
        for term in (
            f"{name}.shape[{dim}] > {int32_max}ULL",
            f"{name}.strides[{dim}] > {int32_max}LL",
            f"{name}.strides[{dim}] < {int32_min}LL",
        )
    )


class Handle:
    def __init__(self, compilation, kernel, launch, source, library=None):
        self._compilation = compilation
        self._artifact = compilation.artifact
        self._backend = compilation.artifact.backend.value
        self._kernel = kernel
        self._launch = launch
        self._source = str(source)
        self._library = None if library is None else str(library)
        self._ssa = compilation.kernel.ssa
        self._pass_trace = compilation.pass_trace
        self._launch_plan = compilation.launch_plan
        cache_key = compilation_cache_key(compilation)
        manifest = (
            Path(self._library).with_suffix(".manifest.json")
            if self._library is not None
            else Path(self._source).with_suffix(".manifest.json")
        )
        write_manifest(
            manifest,
            _built_manifest(
                compilation,
                cache_key,
                Path(self._source),
                None if self._library is None else Path(self._library),
            ),
        )
        self._built_artifact = BuiltArtifact(
            source=compilation.artifact,
            cache_key=cache_key,
            source_path=self._source,
            binary_path=self._library,
            manifest_path=str(manifest),
            abi=compilation.artifact.metadata.get("launch_abi", {}),
        )

    def __call__(self, *args, **kwargs):
        return self._launch(*args, **kwargs)


def materialize(
    compilation,
    *,
    output_dir: str | Path | None = None,
    mode: str = "jit",
) -> Handle:
    target = compilation.artifact.backend

    if target != Target.TRITON and _requires_runtime_specialization(compilation):
        return _materialize_lazy(compilation, output_dir=output_dir, mode=mode)

    from ninetoothed.backends.materializers import materializer_for

    materializer = materializer_for(target)

    if mode == "jit":
        return materializer.jit_materialize(compilation, output_dir=output_dir)

    if mode == "aot":
        if output_dir is None:
            raise ValueError("AOT materialization requires an output directory.")
        return materializer.aot_build(compilation, output_dir=output_dir)

    raise ValueError(f"Unknown materialization mode `{mode}`.")


def load_built_artifact(built: BuiltArtifact):
    """Load a materialized binary and restore its public Launch ABI callable."""
    from ninetoothed.backends.materializers import materializer_for

    return materializer_for(built.source.backend).load_built_artifact(built)


def _launch_abi_from_dict(value) -> LaunchABI:
    return LaunchABI(
        public_args=tuple(value.get("public_args", ())),
        kernel_args=tuple(
            LaunchBinding(**dict(binding)) for binding in value.get("kernel_args", ())
        ),
        outputs=tuple(value.get("outputs", ())),
        shape_params=tuple(value.get("shape_params", ())),
    )


def _runtime_specs(artifact):
    return tuple(
        SimpleNamespace(
            name=str(value["name"]),
            ndim=int(value.get("ndim", 0)),
            dtype=value.get("dtype"),
            attrs=dict(value.get("attrs", {})),
        )
        for value in artifact.metadata.get("tensors", ())
    )


def _requires_runtime_specialization(compilation) -> bool:
    if any(tensor.dtype is None for tensor in compilation.kernel.tensors):
        return True

    if compilation.request.specialization_values:
        return False
    return compilation.artifact.backend == Target.TILELANG and any(
        binding.kind
        in {
            "shape",
            "stride",
            "meta",
            "jagged_values_numel",
            "jagged_offsets_numel",
        }
        for binding in compilation.launch_abi.kernel_args
    )


def _materialize_lazy(compilation, *, output_dir=None, mode="jit") -> Handle:
    artifact = compilation.artifact
    suffix = {
        Target.CUDA: "cu",
        Target.TILELANG: "tilelang.py",
    }[artifact.backend]
    cache_key = compilation_cache_key(compilation)
    source = write_source(
        artifact.kernel_name,
        artifact.primary_source,
        suffix,
        cache_key=cache_key,
    )
    specializations: dict[tuple[tuple[str, str], ...], Handle] = {}
    handle = None

    def launch(*args, **kwargs):
        nonlocal handle
        public = _public_values(
            compilation.launch_abi,
            args,
            kwargs,
            specs=compilation.kernel.tensors,
        )
        dtypes = _runtime_dtypes(compilation, public)
        specialization_values = _runtime_specialization_values(compilation, public)
        key = (
            *(f"dtype:{name}={value}" for name, value in sorted(dtypes.items())),
            *(
                f"symbol:{name}={value!r}"
                for name, value in sorted(specialization_values.items())
            ),
        )

        if key not in specializations:
            from ninetoothed.compiler import compile_kernel

            request = replace(
                compilation.request,
                tensors=copy.deepcopy(compilation.request.tensors),
                tensor_dtypes=dtypes,
                specialization_values=specialization_values,
            )
            specializations[key] = materialize(
                compile_kernel(request), output_dir=output_dir, mode=mode
            )

        specialized = specializations[key]

        if handle is not None:
            handle._artifact = specialized._artifact
            handle._kernel = specialized._kernel
            handle._source = specialized._source
            handle._library = specialized._library
            handle._ssa = specialized._ssa
            handle._pass_trace = specialized._pass_trace
            handle._launch_plan = specialized._launch_plan
            handle._built_artifact = specialized._built_artifact
        return specialized(
            *args,
            **_filter_runtime_kwargs(specialized._compilation.launch_abi, kwargs),
        )

    handle = Handle(compilation, None, launch, source)

    return handle


def _runtime_specialization_values(compilation, public) -> dict[str, Any]:
    values = {}
    dynamic_parameters = set(compilation.launch_plan.dynamic_parameters)

    for binding in compilation.launch_abi.kernel_args:
        if binding.name not in dynamic_parameters:
            continue

        value = _binding_value(binding, public)

        if hasattr(value, "item"):
            value = value.item()

        values[binding.name] = value
    return values


def _runtime_dtypes(compilation, public) -> dict[str, str]:
    result = {}

    for tensor in compilation.kernel.tensors:
        if tensor.dtype is not None or tensor.name not in public:
            continue

        value = public[tensor.name]
        dtype = getattr(value, "dtype", None)

        if dtype is not None:
            result[tensor.name] = str(dtype).split(".")[-1]
        elif isinstance(value, bool):
            result[tensor.name] = "bool"
        elif isinstance(value, int):
            result[tensor.name] = "int64"
        elif isinstance(value, float):
            result[tensor.name] = "float32"
    return result


def _is_cacheable_runtime_literal(value) -> bool:
    return (
        value is None
        or type(value) in (bool, int, float, complex, str, bytes)
        or (
            isinstance(value, tuple)
            and all(_is_cacheable_runtime_literal(item) for item in value)
        )
    )


@dataclass(frozen=True, kw_only=True)
class _RuntimeValueContract:
    is_tensor: bool
    identity: int
    value_type: type
    value: Any = None
    shape: tuple | None = None
    stride: tuple | None = None
    dtype: Any = None
    device: Any = None
    data_ptr: int | None = None
    storage_offset: int | None = None
    scalar_value: tuple[type, Any] | None = None
    stride_callable: bool = False
    data_ptr_callable: bool = False
    storage_offset_callable: bool = False
    tensor_state: tuple | None = None

    @property
    def cacheable(self) -> bool:
        return self.is_tensor or _is_cacheable_runtime_literal(self.value)

    @classmethod
    def from_value(cls, value, *, compare_scalar_tensor=False):
        shape = getattr(value, "shape", None)

        if shape is None or not hasattr(value, "dtype"):
            return cls(
                is_tensor=False,
                identity=0,
                value_type=type(value),
                value=value,
            )

        stride = getattr(value, "stride", None)
        data_ptr = getattr(value, "data_ptr", None)
        storage_offset = getattr(value, "storage_offset", None)
        shape = tuple(shape)
        stride_callable = callable(stride)
        data_ptr_callable = callable(data_ptr)
        storage_offset_callable = callable(storage_offset)
        stride_value = tuple(stride()) if stride_callable else None
        device = getattr(value, "device", None)
        data_ptr_value = data_ptr() if data_ptr_callable else None
        storage_offset_value = storage_offset() if storage_offset_callable else None
        scalar_value = None

        if compare_scalar_tensor and not shape:
            item = value.item()
            scalar_value = (type(item), item)

        return cls(
            is_tensor=True,
            identity=id(value),
            value_type=type(value),
            shape=shape,
            stride=stride_value,
            dtype=value.dtype,
            device=device,
            data_ptr=data_ptr_value,
            storage_offset=storage_offset_value,
            scalar_value=scalar_value,
            stride_callable=stride_callable,
            data_ptr_callable=data_ptr_callable,
            storage_offset_callable=storage_offset_callable,
            tensor_state=(
                shape,
                stride_value,
                value.dtype,
                device,
                data_ptr_value,
                storage_offset_value,
            ),
        )

    def matches(self, value, *, identity_verified=False) -> bool:
        if not self.is_tensor:
            return type(value) is self.value_type and value == self.value

        if not identity_verified and (
            id(value) != self.identity or type(value) is not self.value_type
        ):
            return False

        current_state = (
            getattr(value, "shape", None),
            value.stride() if self.stride_callable else None,
            getattr(value, "dtype", None),
            getattr(value, "device", None),
            value.data_ptr() if self.data_ptr_callable else None,
            value.storage_offset() if self.storage_offset_callable else None,
        )

        if current_state != self.tensor_state:
            return False

        if self.scalar_value is None:
            return True

        item = value.item()

        return type(item) is self.scalar_value[0] and item == self.scalar_value[1]


@dataclass(frozen=True, kw_only=True)
class _VerifiedRuntimeCall:
    positional: tuple[_RuntimeValueContract, ...]
    keywords: tuple[tuple[str, _RuntimeValueContract], ...]
    two_tensor_state: tuple | None = None

    @property
    def cacheable(self) -> bool:
        return all(contract.cacheable for contract in self.positional) and all(
            contract.cacheable for _name, contract in self.keywords
        )

    @classmethod
    def from_call(cls, abi, args, kwargs):
        scalar_sources = {
            binding.source
            for binding in abi.kernel_args
            if binding.kind in {"scalar", "constexpr", "meta"}
        }
        positional_names = abi.public_args[: len(args)]
        positional = tuple(
            _RuntimeValueContract.from_value(
                value,
                compare_scalar_tensor=name in scalar_sources,
            )
            for name, value in zip(positional_names, args)
        )
        keywords = tuple(
            (
                name,
                _RuntimeValueContract.from_value(
                    value,
                    compare_scalar_tensor=name in scalar_sources,
                ),
            )
            for name, value in kwargs.items()
        )
        two_tensor_state = None

        if (
            len(positional) == 2
            and all(
                contract.is_tensor
                and contract.stride_callable
                and contract.data_ptr_callable
                and contract.storage_offset_callable
                for contract in positional
            )
            and len(keywords) == 1
            and not keywords[0][1].is_tensor
        ):
            scalar = keywords[0][1]

            try:
                hash(scalar.value)
            except TypeError:
                pass
            else:
                two_tensor_state = (
                    keywords[0][0],
                    scalar.value_type,
                    scalar.value,
                    positional[0].identity,
                    positional[0].value_type,
                    *positional[0].tensor_state,
                    positional[1].identity,
                    positional[1].value_type,
                    *positional[1].tensor_state,
                )
        return cls(
            positional=positional,
            keywords=keywords,
            two_tensor_state=two_tensor_state,
        )

    def call_key(self, args, kwargs):
        if self.two_tensor_state is None or len(args) != 2 or len(kwargs) != 1:
            return None

        expected_name = self.keywords[0][0]

        try:
            scalar = kwargs[expected_name]
        except KeyError:
            return None

        if type(scalar) is not self.keywords[0][1].value_type:
            return None

        first, second = args

        try:
            return (
                expected_name,
                type(scalar),
                scalar,
                id(first),
                type(first),
                first.shape,
                first.stride(),
                first.dtype,
                first.device,
                first.data_ptr(),
                first.storage_offset(),
                id(second),
                type(second),
                second.shape,
                second.stride(),
                second.dtype,
                second.device,
                second.data_ptr(),
                second.storage_offset(),
            )
        except (AttributeError, RuntimeError, TypeError):
            return None

    def matches(
        self,
        args,
        kwargs,
        *,
        identity_verified=False,
        call_key=None,
    ) -> bool:
        if len(args) != len(self.positional) or len(kwargs) != len(self.keywords):
            return False

        if self.two_tensor_state is not None:
            if call_key is None:
                call_key = self.call_key(args, kwargs)
            return call_key == self.two_tensor_state

        for contract, value in zip(self.positional, args):
            if not contract.matches(value, identity_verified=identity_verified):
                return False

        for (name, value), (expected_name, contract) in zip(
            kwargs.items(), self.keywords
        ):
            if name != expected_name or not contract.matches(
                value,
                identity_verified=identity_verified,
            ):
                return False
        return True


def _jagged_tensor_state(value):
    try:
        return (
            type(value),
            value.shape,
            value.stride(),
            value.dtype,
            value.device,
            value.data_ptr(),
            value.storage_offset(),
            getattr(value, "_version", None),
        )
    except (AttributeError, RuntimeError, TypeError):
        return None


@dataclass(frozen=True, kw_only=True)
class _JaggedOwnerContract:
    owner: weakref.ReferenceType
    values_state: tuple
    offsets_state: tuple

    @classmethod
    def from_owner(cls, owner):
        try:
            reference = weakref.ref(owner)
            values_state = _jagged_tensor_state(owner.values())
            offsets_state = _jagged_tensor_state(owner.offsets())
        except (AttributeError, RuntimeError, TypeError):
            return None

        if values_state is None or offsets_state is None:
            return None

        return cls(
            owner=reference,
            values_state=values_state,
            offsets_state=offsets_state,
        )

    def matches(self):
        owner = self.owner()

        if owner is None:
            return False

        try:
            state = (
                _jagged_tensor_state(owner.values()),
                _jagged_tensor_state(owner.offsets()),
            )
        except (AttributeError, RuntimeError, TypeError):
            return False

        return state == (self.values_state, self.offsets_state)


@dataclass(frozen=True, kw_only=True)
class _PreparedBoundValue:
    value: Any = None
    tensor: weakref.ReferenceType | None = None
    owner: weakref.ReferenceType | None = None
    binding: LaunchBinding | None = None
    flatten_tensor: bool = False

    @classmethod
    def from_value(cls, binding, value, public, *, flatten_tensors):
        if binding.kind.startswith("jagged_"):
            try:
                owner = weakref.ref(public[binding.source])
            except (KeyError, TypeError):
                return None

            return cls(
                owner=owner,
                binding=binding,
                flatten_tensor=flatten_tensors
                and binding.kind in {"jagged_values", "jagged_offsets"},
            )

        is_tensor = hasattr(value, "shape") and hasattr(value, "dtype")

        if binding.kind != "tensor":
            return cls(value=value) if _is_cacheable_runtime_literal(value) else None

        if not is_tensor:
            return None

        try:
            tensor = weakref.ref(value)
        except TypeError:
            return None

        return cls(tensor=tensor, flatten_tensor=flatten_tensors)

    def resolve(self, *, flatten_tensor=False):
        if self.binding is not None:
            owner = self.owner()

            if owner is None:
                return None, None

            value = _jagged_binding_value(self.binding, owner)
        elif self.tensor is not None:
            value = self.tensor()
        else:
            return self.value, None

        if value is None:
            return None, None

        if (
            not self.flatten_tensor
            or not flatten_tensor
            or not hasattr(value, "as_strided")
        ):
            return value, None

        storage_numel = value.untyped_storage().nbytes() // value.element_size()
        storage_offset = value.storage_offset()
        flattened = value.as_strided(
            (storage_numel - storage_offset,),
            (1,),
            storage_offset=storage_offset,
        )

        return flattened, flattened


@dataclass(frozen=True, kw_only=True)
class _RuntimeBindingPlan:
    values: tuple[_PreparedBoundValue, ...]
    jagged_owners: tuple[_JaggedOwnerContract, ...]

    @classmethod
    def from_values(cls, bindings, values, public, *, flatten_tensors):
        planned = tuple(
            _PreparedBoundValue.from_value(
                binding,
                value,
                public,
                flatten_tensors=flatten_tensors,
            )
            for binding, value in zip(bindings, values)
        )

        if any(value is None for value in planned):
            return None

        jagged_owners = []
        owner_ids = set()

        for value in planned:
            if value.owner is None:
                continue

            owner = value.owner()

            if owner is None:
                return None

            identity = id(owner)

            if identity in owner_ids:
                continue

            contract = _JaggedOwnerContract.from_owner(owner)

            if contract is None:
                return None

            owner_ids.add(identity)
            jagged_owners.append(contract)

        return cls(values=planned, jagged_owners=tuple(jagged_owners))

    @property
    def owner_refs(self):
        return tuple(
            owner
            for value in self.values
            for owner in (value.tensor, value.owner)
            if owner is not None
        )

    def static_values(self):
        values = []

        for planned in self.values:
            if planned.binding is not None:
                return None

            if planned.tensor is None:
                values.append(planned.value)
                continue

            value = planned.tensor()

            if value is None:
                return None

            values.append(None)

        return tuple(values)

    def call_sources(self, bindings, args, kwargs, public_args):
        positions = {name: index for index, name in enumerate(public_args[: len(args)])}
        sources = []

        for binding, planned in zip(bindings, self.values):
            if planned.tensor is None:
                sources.append(None)
            elif binding.source in positions:
                sources.append(("positional", positions[binding.source]))
            elif binding.source in kwargs:
                sources.append(("keyword", binding.source))
            else:
                return None

        return tuple(sources)

    def resolve(self, *, flatten_tensors=False):
        values = []
        keepalive = []

        for planned in self.values:
            value, temporary = planned.resolve(flatten_tensor=flatten_tensors)

            if (
                planned.tensor is not None or planned.owner is not None
            ) and value is None:
                return None

            values.append(value)

            if temporary is not None:
                keepalive.append(temporary)

        return tuple(values), tuple(keepalive)


@dataclass(frozen=True, kw_only=True)
class _PreparedRuntimeLaunch:
    guard: _VerifiedRuntimeCall
    binding_plan: _RuntimeBindingPlan | None
    owner_refs: tuple[weakref.ReferenceType, ...] | None
    empty: bool
    invocation_plan: Callable[[tuple, tuple, dict], Any] | None = None
    cache_token: object | None = None

    @property
    def cacheable(self) -> bool:
        return (
            self.guard.cacheable
            and self.owner_refs is not None
            and (self.empty or self.binding_plan is not None)
        )

    def matches(self, args, kwargs, *, identity_verified=False, call_key=None):
        if not self.guard.matches(
            args,
            kwargs,
            identity_verified=identity_verified,
            call_key=call_key,
        ):
            return False

        return (
            self.binding_plan is None
            or not self.binding_plan.jagged_owners
            or all(contract.matches() for contract in self.binding_plan.jagged_owners)
        )


_VERIFIED_RUNTIME_CALL_CACHE_SIZE = 8


def _runtime_call_identity(args, kwargs):
    return (len(args), *map(id, args), *map(id, kwargs.values()))


def _two_tensor_call_identity(call_key):
    return (call_key[3], call_key[11], call_key[1], call_key[2])


def _remember_verified_runtime_call(prepared_calls, identity, prepared):
    prepared_calls.pop(identity, None)

    if len(prepared_calls) >= _VERIFIED_RUNTIME_CALL_CACHE_SIZE:
        prepared_calls.pop(next(iter(prepared_calls)))

    prepared_calls[identity] = prepared


def _runtime_owner_refs(args, kwargs, binding_plan=None):
    refs = []
    identities = set()

    def append(owner, reference=None):
        identity = id(owner)

        if identity in identities:
            return True

        try:
            reference = reference or weakref.ref(owner)
        except TypeError:
            return False

        identities.add(identity)
        refs.append(reference)

        return True

    for value in (*args, *kwargs.values()):
        if getattr(value, "shape", None) is None or not hasattr(value, "dtype"):
            continue

        if not append(value):
            return None

    if binding_plan is not None:
        for reference in binding_plan.owner_refs:
            owner = reference()

            if owner is None or not append(owner, reference):
                return None

    return tuple(refs)


def _arm_prepared_runtime_launch(prepared, callback, token):
    if not prepared.cacheable:
        return None

    owners = tuple(owner() for owner in prepared.owner_refs)

    if any(owner is None for owner in owners):
        return None

    return replace(
        prepared,
        owner_refs=tuple(weakref.ref(owner, callback) for owner in owners),
        cache_token=token,
    )


def _verified_runtime_launch(launch):
    active_identity = None
    active = None
    prepared_calls = {}

    def evict(identity, token):
        nonlocal active, active_identity
        cached = prepared_calls.get(identity)

        if cached is not None and cached.cache_token is token:
            prepared_calls.pop(identity, None)

        if active is not None and active.cache_token is token:
            active = None
            active_identity = None

    def activate(identity, prepared):
        nonlocal active, active_identity
        active_identity = identity
        active = prepared

    def remember(identity, prepared):
        token = object()

        def collected(_reference):
            evict(identity, token)

        prepared = _arm_prepared_runtime_launch(prepared, collected, token)

        if prepared is not None:
            _remember_verified_runtime_call(prepared_calls, identity, prepared)
            activate(identity, prepared)

        return prepared

    def verified(*args, **kwargs):
        active_snapshot = active
        active_identity_snapshot = active_identity
        call_key = (
            active_snapshot.guard.call_key(args, kwargs)
            if active_snapshot is not None
            and active_snapshot.guard.two_tensor_state is not None
            else None
        )
        identity = (
            _two_tensor_call_identity(call_key)
            if call_key is not None
            else _runtime_call_identity(args, kwargs)
        )

        if (
            active_snapshot is not None
            and identity == active_identity_snapshot
            and active_snapshot.matches(
                args,
                kwargs,
                identity_verified=True,
                call_key=call_key,
            )
        ):
            return launch._ninetoothed_invoke_prepared(
                active_snapshot,
                args,
                kwargs,
            )

        cached = prepared_calls.pop(identity, None)

        if cached is not None and cached.matches(
            args,
            kwargs,
            identity_verified=True,
            call_key=call_key,
        ):
            prepared_calls[identity] = cached
            activate(identity, cached)

            return launch._ninetoothed_invoke_prepared(cached, args, kwargs)

        prepared = launch._ninetoothed_prepare(args, kwargs)
        cached_prepared = remember(
            _two_tensor_call_identity(prepared.guard.two_tensor_state)
            if prepared.guard.two_tensor_state is not None
            else identity,
            prepared,
        )

        return launch._ninetoothed_invoke_prepared(
            cached_prepared or prepared,
            args,
            kwargs,
        )

    return verified


def _runtime_wrapper(
    function,
    abi: LaunchABI,
    *,
    low_level: bool = True,
    specs=(),
    binding_overrides=None,
    prepare_invocation=None,
):
    overrides = dict(binding_overrides or {})

    def prepare(args, kwargs, *, public=None):
        if public is None:
            public = _public_values(abi, args, kwargs, specs=specs)

        bound_public = dict(public) | overrides

        if _empty_launch(abi, public):
            return _PreparedRuntimeLaunch(
                guard=_VerifiedRuntimeCall.from_call(abi, args, kwargs),
                binding_plan=None,
                owner_refs=_runtime_owner_refs(args, kwargs),
                empty=True,
            )

        values, _keepalive = _bound_values(abi, bound_public, scalar_mode="value")
        bindings = abi.kernel_args

        binding_plan = _RuntimeBindingPlan.from_values(
            bindings,
            values,
            bound_public,
            flatten_tensors=low_level,
        )
        resolved = (
            binding_plan.resolve(flatten_tensors=low_level)
            if binding_plan is not None
            else None
        )
        resolved_values = resolved[0] if resolved is not None else ()
        static_values = (
            binding_plan.static_values() if binding_plan is not None else None
        )
        call_sources = (
            binding_plan.call_sources(
                bindings,
                args,
                kwargs,
                abi.public_args,
            )
            if binding_plan is not None and static_values is not None
            else None
        )

        if static_values is not None and call_sources is None:
            static_values = None

        invocation_plan = (
            prepare_invocation(resolved_values, static_values, call_sources)
            if prepare_invocation is not None and resolved is not None
            else None
        )
        owner_refs = _runtime_owner_refs(args, kwargs, binding_plan)

        return _PreparedRuntimeLaunch(
            guard=_VerifiedRuntimeCall.from_call(abi, args, kwargs),
            binding_plan=binding_plan,
            owner_refs=owner_refs,
            empty=False,
            invocation_plan=invocation_plan,
        )

    def invoke(prepared, args, kwargs):
        if prepared.empty:
            return _first_output_from_call(abi, args, kwargs)

        invocation_plan = prepared.invocation_plan
        static_invocation = invocation_plan is not None and not getattr(
            invocation_plan,
            "requires_values",
            True,
        )
        resolved = (
            ((), ())
            if static_invocation
            else (
                prepared.binding_plan.resolve(flatten_tensors=low_level)
                if prepared.binding_plan is not None
                else None
            )
        )

        if resolved is None:
            return launch(*args, **kwargs)

        values, keepalive = resolved

        if invocation_plan is not None:
            result = invocation_plan(values, args, kwargs)
        else:
            result = function(*values) if low_level else function(*args, **kwargs)

        del keepalive

        if result is not None and not abi.outputs:
            return result
        return _first_output_from_call(abi, args, kwargs)

    def launch(*args, **kwargs):
        public = _public_values(abi, args, kwargs, specs=specs)

        if _empty_launch(abi, public):
            return _first_output(abi, public)

        values, keepalive = _bound_values(
            abi,
            dict(public) | overrides,
            scalar_mode="value",
        )

        if low_level:
            values, flattened = _flatten_ffi_tensor_args(abi.kernel_args, values)
            keepalive.extend(flattened)

        result = function(*values) if low_level else function(*args, **kwargs)
        del keepalive

        if result is not None and not abi.outputs:
            return result
        return _first_output(abi, public)

    launch._ninetoothed_prepare = prepare
    launch._ninetoothed_invoke_prepared = invoke

    return launch


def _public_values(abi: LaunchABI, args, kwargs, *, specs=()) -> dict[str, Any]:
    if len(args) > len(abi.public_args):
        raise TypeError(f"Expected at most {len(abi.public_args)} arguments.")

    positional_names = abi.public_args[: len(args)]
    duplicates = tuple(name for name in positional_names if name in kwargs)

    if duplicates:
        raise TypeError(
            f"Kernel arguments passed twice: {', '.join(sorted(duplicates))}."
        )

    accepted = set(abi.public_args) | {
        binding.source
        for binding in abi.kernel_args
        if binding.kind == "meta" and binding.source is not None
    }
    unknown = tuple(name for name in kwargs if name not in accepted)

    if unknown:
        raise TypeError(f"Unknown kernel arguments: {', '.join(sorted(unknown))}.")

    values = dict(zip(abi.public_args, args))
    values.update(kwargs)
    missing = tuple(name for name in abi.public_args if name not in values)

    if missing:
        raise TypeError(f"Missing kernel arguments: {', '.join(missing)}.")

    _validate_runtime_values(values, specs)

    return values


def _filter_runtime_kwargs(abi: LaunchABI, kwargs) -> dict[str, Any]:
    accepted = set(abi.public_args) | {
        binding.source
        for binding in abi.kernel_args
        if binding.kind == "meta" and binding.source is not None
    }

    return {name: value for name, value in kwargs.items() if name in accepted}


def _validate_runtime_values(values, specs) -> None:
    expected_device = None

    for spec in specs:
        if spec.name not in values:
            continue

        value = values[spec.name]
        expected_device = _validate_tensor_contract(spec, value, expected_device)
        _validate_dtype_contract(spec, value)


def _validate_tensor_contract(spec, value, expected_device):
    source_ndim = int(spec.attrs.get("source_ndim", spec.ndim))

    if source_ndim == 0:
        return expected_device

    shape = getattr(value, "shape", None)

    if shape is None:
        raise TypeError(f"Kernel argument `{spec.name}` must be a tensor.")

    if len(shape) != source_ndim:
        raise TypeError(
            f"Kernel argument `{spec.name}` has rank {len(shape)}; "
            f"expected {source_ndim}."
        )

    device = getattr(value, "device", None)

    if device is None:
        return expected_device

    device_type = getattr(device, "type", str(device).split(":")[0])

    if device_type != "cuda":
        raise TypeError(f"Kernel argument `{spec.name}` must be on a CUDA device.")

    if expected_device is not None and device != expected_device:
        raise TypeError("All tensor arguments must use the same CUDA device.")
    return device if expected_device is None else expected_device


def _validate_dtype_contract(spec, value) -> None:
    source_dtype = spec.attrs.get("source_dtype", spec.dtype)

    if source_dtype is None or not hasattr(value, "dtype"):
        return

    actual = str(value.dtype).split(".")[-1]
    expected = _canonical_dtype(source_dtype)

    if actual != expected:
        raise TypeError(
            f"Kernel argument `{spec.name}` has dtype {actual}; expected {expected}."
        )


def _canonical_dtype(dtype) -> str:
    name = str(dtype).split(".")[-1]

    return {
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "bf16": "bfloat16",
    }.get(name, name)


def _bound_values(abi, public, *, scalar_mode, specs=None, cuda_scalar=None):
    values = []
    keepalive = []

    for binding in abi.kernel_args:
        value = _binding_value(binding, public)

        if binding.kind in {"tensor", "jagged_values", "jagged_offsets"}:
            if scalar_mode == "cuda":
                value = ctypes.c_void_p(value.data_ptr())
        elif binding.kind in {"scalar", "constexpr"}:
            if hasattr(value, "item"):
                value = value.item()

            if scalar_mode == "cuda":
                spec = specs.get(binding.source) if specs is not None else None

                if spec is not None and cuda_scalar is None:
                    raise RuntimeError(
                        "CUDA scalar binding requires a dtype converter."
                    )

                value = (
                    cuda_scalar(value, spec.dtype)
                    if spec is not None
                    else ctypes.c_int64(int(value))
                )
        elif scalar_mode == "cuda":
            value = ctypes.c_int64(int(value))

        values.append(value)
    return values, keepalive


def _flatten_ffi_tensor_args(bindings, values):
    flattened = []
    result = []

    for binding, value in zip(bindings, values):
        if binding.kind in {
            "tensor",
            "jagged_values",
            "jagged_offsets",
        } and hasattr(value, "as_strided"):
            storage_numel = value.untyped_storage().nbytes() // value.element_size()
            storage_offset = value.storage_offset()
            value = value.as_strided(
                (storage_numel - storage_offset,),
                (1,),
                storage_offset=storage_offset,
            )
            flattened.append(value)

        result.append(value)
    return result, flattened


def _binding_value(binding: LaunchBinding, public):
    if binding.kind in {"tensor", "scalar", "constexpr"}:
        return public[binding.source]

    if binding.kind.startswith("jagged_"):
        return _jagged_binding_value(binding, public[binding.source])

    if binding.kind == "shape":
        return public[binding.source].shape[binding.dim]

    if binding.kind == "stride":
        return public[binding.source].stride(binding.dim)

    if binding.kind == "meta":
        if binding.source in public:
            return public[binding.source]

        if binding.value is not None:
            return binding.value

        raise TypeError(f"Missing launch meta-parameter `{binding.name}`.")
    return binding.value


def _jagged_binding_value(binding: LaunchBinding, value):
    accessors = {
        "jagged_values": lambda: value.values(),
        "jagged_offsets": lambda: value.offsets(),
        "jagged_values_numel": lambda: value.values().numel(),
        "jagged_offsets_numel": lambda: value.offsets().numel(),
        "jagged_max_seq_len": lambda: value.offsets().diff().max().item(),
    }

    try:
        return accessors[binding.kind]()
    except KeyError:
        return binding.value


def _first_output(abi, public):
    return public[abi.outputs[0]] if abi.outputs else None


def _first_output_from_call(abi, args, kwargs):
    if not abi.outputs:
        return None

    name = abi.outputs[0]
    index = abi.public_args.index(name)

    return args[index] if index < len(args) else kwargs[name]


def _empty_launch(abi, public) -> bool:
    names = abi.outputs or tuple(
        binding.source
        for binding in abi.kernel_args
        if binding.kind in {"tensor", "jagged_values", "jagged_offsets"}
        and binding.source is not None
    )

    return any(
        name in public and hasattr(public[name], "numel") and public[name].numel() == 0
        for name in names
    )


def import_python_module(path: str | Path, module_name: str | None = None):
    """Import a generated Python module from an artifact path."""
    path = Path(path)
    module_name = module_name or f"_ninetoothed_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import generated artifact `{path}`.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def _export_library_atomic(module, library: Path) -> None:
    temporary = _temporary_output(library)

    try:
        module.export_library(str(temporary))
        os.replace(temporary, library)
    finally:
        temporary.unlink(missing_ok=True)


def _temporary_output(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix,
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()

    return temporary


def _replace_file(source: Path, destination: Path) -> None:
    temporary = _temporary_output(destination)

    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_library(cache_library: Path, output_dir, filename: str) -> Path:
    if output_dir is None:
        return cache_library

    destination = Path(output_dir) / filename
    atomic_write_bytes(destination, cache_library.read_bytes())

    return destination


def _built_manifest(compilation, cache_key, source, library):
    return {
        "schema": 2,
        "cache_key": cache_key,
        "backend": compilation.artifact.backend.value,
        "kernel_name": compilation.artifact.kernel_name,
        "entrypoint": compilation.artifact.entrypoint,
        "source": str(source),
        "library": None if library is None else str(library),
        "launch_abi": compilation.artifact.metadata.get("launch_abi", {}),
        "launch_plan": ir_to_dict(compilation.launch_plan),
        "pass_trace": compilation.pass_trace,
    }


__all__ = [
    "Handle",
    "KernelLaunchError",
    "import_python_module",
    "load_built_artifact",
    "materialize",
    "overflow_terms",
]
