"""Execute pure SSA partitions through injected ordinary native stage handles."""

import hashlib
from dataclasses import replace
from pathlib import Path

from ninetoothed.backends import emit
from ninetoothed.compiler.cache import compilation_cache_key, write_source
from ninetoothed.compiler.driver import Compilation
from ninetoothed.compiler.materialization import (
    dense_scratch_spec,
)
from ninetoothed.dtype import normalize_dtype
from ninetoothed.ir import IndexExpr, LaunchABI, LaunchBinding, LaunchPlan, ir_to_dict


def _static_dense_contract(spec):
    if spec.dtype is None or spec.jagged_dim is not None:
        return False

    if spec.attrs.get("jagged_dim") is not None:
        return False

    rank = int(spec.attrs.get("source_ndim", spec.ndim))
    shape = spec.attrs.get("source_shape", ())

    if rank != len(shape) or spec.attrs.get("source_dtype", spec.dtype) is None:
        return False

    try:
        return all(int(dim) > 0 and str(int(dim)) == str(dim) for dim in shape)
    except (TypeError, ValueError):
        return False


def _derived_native_stage_launch(stage, handle, originals, scratch, device_type):
    """Reuse outer contracts only for identical dense specs and standard ABIs."""
    from ninetoothed.backends.materializers.native import _NativeTransientLaunch

    native = getattr(
        getattr(handle, "_launch", None), "_ninetoothed_native_transient", None
    )

    if (
        not isinstance(native, _NativeTransientLaunch)
        or getattr(handle, "_compilation", None) is not stage
        or native.abi != stage.launch_abi
        or native.tensor_specs != stage.kernel.tensors
        or native.device_types != (device_type,)
        or native.module != device_type
        or {"cuda": "cuda", "bangc": "mlu", "ascendc": "npu"}.get(
            stage.artifact.backend.value
        )
        != device_type
        or native.stream_attr
        != {"cuda": "cuda_stream", "mlu": "mlu_stream", "npu": "npu_stream"}[
            device_type
        ]
        or stage.launch_abi.public_args
        != tuple(spec.name for spec in stage.kernel.tensors)
        or stage.artifact.metadata.get("launch_abi") != ir_to_dict(stage.launch_abi)
    ):
        return None

    specs = {spec.name: spec for spec in stage.kernel.tensors}

    if len(specs) != len(stage.kernel.tensors):
        return None

    for name, spec in specs.items():
        if spec != (originals | scratch).get(name) or not _static_dense_contract(spec):
            return None

    for binding in stage.launch_abi.kernel_args:
        spec = specs.get(binding.source)

        if spec is None:
            return None

        rank = int(spec.attrs.get("source_ndim", spec.ndim))

        if binding.kind == "tensor":
            if rank == 0:
                return None
        elif binding.kind in {"shape", "stride"}:
            if not isinstance(binding.dim, int) or not 0 <= binding.dim < rank:
                return None
        elif binding.kind in {"scalar", "constexpr"}:
            if rank != 0:
                return None
        else:
            return None
    return native


def _allocated_scratch_contracts(plan):
    expected = getattr(plan, "scratch_specs", None)
    specs = (
        tuple(expected)
        if expected is not None
        else tuple(dense_scratch_spec(tensor) for tensor in plan.scratch)
    )
    by_name = {spec.name: spec for spec in specs}

    if len(by_name) != len(specs) or set(by_name) != {
        tensor.name for tensor in plan.scratch
    }:
        return None

    for tensor in plan.scratch:
        spec = by_name[tensor.name]

        if (
            not _static_dense_contract(spec)
            or int(spec.attrs.get("source_ndim", spec.ndim)) != len(tensor.shape)
            or tuple(map(str, spec.attrs.get("source_shape", ())))
            != tuple(map(str, tensor.shape))
            or normalize_dtype(str(spec.attrs.get("source_dtype", spec.dtype)))
            != normalize_dtype(tensor.dtype)
            or normalize_dtype(str(spec.dtype)) != normalize_dtype(tensor.dtype)
        ):
            return None
    return by_name


def aot_monolithic_compilation(compilation):
    """Keep implicit native staging compatible with the existing AOT source."""
    selection = compilation.artifact.metadata.get("ssa_metadata", {}).get(
        "materialization", {}
    )

    if compilation.artifact.execution_plans.get("materialization") is None:
        return compilation

    backend = compilation.artifact.backend.value

    if selection.get("explicitly_enabled"):
        raise ValueError(
            f"The {backend.upper()} staged SSA materialization currently supports JIT only."
        )

    artifact = replace(
        compilation.artifact,
        execution_plans={
            name: plan
            for name, plan in compilation.artifact.execution_plans.items()
            if name != "materialization"
        },
        metadata={
            **compilation.artifact.metadata,
            "execution_route": f"{backend}-monolithic-aot",
            "primary_source_role": "executed",
            "ssa_metadata": {
                **compilation.artifact.metadata.get("ssa_metadata", {}),
                "materialization": {
                    **selection,
                    "enabled": False,
                    "selected": False,
                    "planned": True,
                    "execution_fallback": "aot-single-kernel",
                },
            },
        },
    )

    return replace(compilation, artifact=artifact)


def stage_compilations(compilation):
    plan = compilation.artifact.execution_plans.get("materialization")

    if plan is None:
        return None
    return stage_compilations_from_plan(compilation, plan)


def stage_compilations_from_plan(compilation, plan, *, scratch_specs=None):
    """Emit a proved partition with ordinary tensor and stride launch bindings."""
    originals = {spec.name: spec for spec in compilation.kernel.tensors}
    scratch = (
        {tensor.name: dense_scratch_spec(tensor) for tensor in plan.scratch}
        if scratch_specs is None
        else {spec.name: spec for spec in scratch_specs}
    )

    if set(scratch) != {tensor.name for tensor in plan.scratch}:
        raise ValueError(
            "Every planned scratch tensor requires exactly one specification."
        )

    original_bindings = {
        binding.name: binding for binding in compilation.launch_abi.kernel_args
    }
    result = []

    for stage in plan.stages:
        specs = tuple(
            (originals | scratch)[value.name] for value in stage.program.inputs
        )
        pass_options = {
            **compilation.kernel.compiler_options.get("ssa_pass_options", {}),
            "ssa.plan_materialization": {"enabled": False},
        }
        options = {
            **compilation.kernel.compiler_options,
            "ssa_pass_options": pass_options,
        }
        kernel = replace(
            compilation.kernel,
            kernel_name=stage.program.kind,
            entrypoint=stage.program.kind,
            tensors=specs,
            ssa=stage.program,
            compiler_options=options,
            launch_abi=None,
            launch_plan=None,
        )
        artifact = emit(
            kernel,
            backend=compilation.target.backend,
            target_context=compilation.target,
        )
        outputs = tuple(artifact.metadata["outputs"])
        bindings = [
            LaunchBinding(
                name=name,
                kind="tensor",
                source=name,
                access="write" if name in outputs else "read",
            )
            for name in (*artifact.metadata["variables"], *outputs)
        ]

        for name in artifact.metadata["shape_params"]:
            if name in original_bindings:
                bindings.append(original_bindings[name])
                continue

            derived = next(
                (
                    LaunchBinding(name=name, kind="stride", source=spec.name, dim=dim)
                    for spec in specs
                    for dim, stride in enumerate(spec.attrs.get("source_strides", ()))
                    if str(stride) == name
                ),
                None,
            )

            if derived is None:
                raise ValueError(
                    f"SSA materialization cannot bind stage parameter {name!r}."
                )

            bindings.append(derived)

        abi = LaunchABI(
            public_args=tuple(spec.name for spec in specs),
            kernel_args=tuple(bindings),
            outputs=outputs,
            shape_params=tuple(artifact.metadata["shape_params"]),
        )
        launch = LaunchPlan(
            abi=abi,
            grid=tuple(
                IndexExpr.parse(value) for value in artifact.metadata["launch_grid"]
            ),
            block=tuple(
                IndexExpr.parse(value) for value in artifact.metadata["launch_block"]
            ),
        )
        kernel = replace(kernel, launch_abi=abi, launch_plan=launch)
        artifact = replace(
            artifact,
            metadata={
                **artifact.metadata,
                "launch_abi": ir_to_dict(abi),
                "launch_plan": ir_to_dict(launch),
                "materialization_stage": stage.program.metadata,
            },
        )
        request = replace(
            compilation.request,
            kernel_name=stage.program.kind,
            pass_options=pass_options,
        )
        result.append(
            Compilation(
                request=request,
                target=compilation.target,
                kernel=kernel,
                artifact=artifact,
                launch_plan=launch,
                pass_trace=tuple(artifact.metadata["ssa_pass_trace"]),
            )
        )
    return plan, tuple(result)


def materialize_staged(
    compilation, *, stage_materializer, device_type, output_dir=None
):
    backend = compilation.artifact.backend.value
    expected_device = {"cuda": "cuda", "bangc": "mlu", "ascendc": "npu"}.get(backend)

    if device_type != expected_device:
        raise ValueError(
            f"SSA stage target {backend!r} requires device {expected_device!r}."
        )

    built = stage_compilations(compilation)

    if built is None:
        return None

    plan, stages = built

    return materialize_stage_sequence(
        compilation,
        plan,
        stages,
        stage_materializer=stage_materializer,
        device_type=device_type,
        output_dir=output_dir,
    )


def materialize_stage_sequence(
    compilation,
    plan,
    stages,
    *,
    stage_materializer,
    device_type,
    output_dir=None,
    argument_guard=None,
    fallback_handle=None,
    _transient_bindings=True,
):
    """Execute proved stages with per-call owners and an optional whole-call guard."""
    from ninetoothed.compiler.runtime import (
        Handle,
        _runtime_wrapper,
        _verified_runtime_launch,
    )

    backend = compilation.artifact.backend.value
    expected_device = {"cuda": "cuda", "bangc": "mlu", "ascendc": "npu"}.get(backend)

    if device_type != expected_device:
        raise ValueError(
            f"SSA stage target {backend!r} requires device {expected_device!r}."
        )

    if (argument_guard is None) != (fallback_handle is None):
        raise ValueError(
            "A guarded stage sequence requires its complete fallback handle."
        )

    handles = tuple(
        stage_materializer(
            stage,
            output_dir=None
            if output_dir is None
            else Path(output_dir) / f"stage-{index}",
        )
        for index, stage in enumerate(stages)
    )
    originals = {spec.name: spec for spec in compilation.kernel.tensors}
    memory_sources = {
        binding.source
        for stage in stages
        for binding in stage.launch_abi.kernel_args
        if binding.kind in {"tensor", "jagged_values", "jagged_offsets"}
    }
    device_argument = next(
        (name for name in compilation.launch_abi.public_args if name in memory_sources),
        None,
    )

    if device_argument is None:
        raise ValueError(
            "SSA stage execution requires a public tensor memory argument."
        )

    memory_sources.update(tensor.name for tensor in plan.scratch)
    scratch = _allocated_scratch_contracts(plan)
    transient = tuple(
        _derived_native_stage_launch(stage, handle, originals, scratch, device_type)
        if scratch is not None and not set(originals) & set(scratch)
        else None
        for stage, handle in zip(stages, handles)
    )
    artifact = replace(
        compilation.artifact,
        metadata={
            **compilation.artifact.metadata,
            "execution_route": f"{backend}-guarded-staged-ssa"
            if argument_guard is not None
            else f"{backend}-staged-ssa",
            "primary_source_role": "whole-call-guard-fallback"
            if argument_guard is not None
            else "monolithic-reference-only",
            "materialization_plan": plan.metadata(),
            "whole_call_guard": argument_guard is not None,
            "stage_binding": {
                "proof": "static-derived-public-and-allocated-scratch-contracts",
                "enabled": bool(_transient_bindings),
                "eligible": tuple(launch is not None for launch in transient),
            },
            "materialized_stages": tuple(
                {
                    "kernel_name": stage.artifact.kernel_name,
                    "source_sha256": hashlib.sha256(
                        stage.artifact.primary_source.encode()
                    ).hexdigest(),
                    "ssa": stage.artifact.metadata["ssa"],
                    "ssa_schedule": stage.artifact.metadata["ssa_schedule"],
                }
                for stage in stages
            ),
        },
    )
    compilation = replace(compilation, artifact=artifact)

    def invoke(*args, **kwargs):
        import torch

        # Composite execution consumes the original public tensors. FFI
        # flattening belongs to each stage's native wrapper, after that
        # stage has checked its own physical tensor rank and strides.
        arguments = dict(zip(compilation.launch_abi.public_args, args))
        arguments.update(kwargs)

        if argument_guard is not None and not argument_guard(arguments):
            return fallback_handle(*args, **kwargs)

        device = arguments[device_argument].device
        runtime = getattr(torch, device_type)

        with runtime.device(device):
            stream = runtime.current_stream(device)
            owners = []

            for tensor in plan.scratch:
                owner = torch.empty(
                    tuple(map(int, tensor.shape)),
                    dtype=getattr(torch, tensor.dtype),
                    device=device,
                )
                owners.append(owner)
                arguments[tensor.name] = owner

            launch_error = None

            try:
                for stage, handle, native in zip(stages, handles, transient):
                    if _transient_bindings and native is not None:
                        native(arguments, stream)
                    else:
                        handle(
                            *(arguments[name] for name in stage.launch_abi.public_args)
                        )
            except BaseException as error:
                launch_error = error
                raise
            finally:
                # Earlier stages may already be enqueued when a later launch
                # fails. Preserve that launch error while recording every owner.
                record_error = None

                for name, tensor in arguments.items():
                    if name not in memory_sources or not hasattr(
                        tensor, "record_stream"
                    ):
                        continue

                    try:
                        tensor.record_stream(stream)
                    except BaseException as error:
                        if record_error is None:
                            record_error = error

                if launch_error is None and record_error is not None:
                    raise record_error
            # All owners survive every stage's asynchronous launch. Record the
            # execution stream as well as the allocator's creation stream.
            # Each invocation allocates fresh owners, including graph capture;
            # there is no device/stream-insensitive reusable workspace cache.

    launch = _runtime_wrapper(
        invoke,
        compilation.launch_abi,
        low_level=False,
        specs=compilation.kernel.tensors,
        device_types=(device_type,),
    )
    launch = _verified_runtime_launch(launch)
    key = compilation_cache_key(compilation)
    source = write_source(
        compilation.artifact.kernel_name,
        compilation.artifact.primary_source,
        Path(compilation.artifact.primary_source_name).suffix.lstrip("."),
        cache_key=key,
    )
    handle = Handle(compilation, handles, launch, source)
    handle._stage_handles = handles
    handle._stage_compilations = stages
    handle._materialization_plan = plan
    # Public immutable sequences describe the kernels actually executed. The
    # primary source remains a useful reference, but is never the launch body.
    handle.stage_artifacts = tuple(stage.artifact for stage in stages)
    handle.stage_kernels = tuple(stage.kernel for stage in stages)

    def execution_artifacts_for(*args, **kwargs):
        arguments = dict(zip(compilation.launch_abi.public_args, args))
        arguments.update(kwargs)

        if argument_guard is not None and not argument_guard(arguments):
            return (fallback_handle._artifact,)
        return handle.stage_artifacts

    handle.execution_artifacts_for = execution_artifacts_for

    return handle


__all__ = [
    "aot_monolithic_compilation",
    "materialize_staged",
    "stage_compilations",
    "stage_compilations_from_plan",
    "materialize_stage_sequence",
]
