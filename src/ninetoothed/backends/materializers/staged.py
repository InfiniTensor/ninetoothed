"""Execute a frozen SSA stage plan using ordinary checked CUDA handles."""

import hashlib
from dataclasses import replace
from pathlib import Path

from ninetoothed.backends import emit
from ninetoothed.compiler.cache import compilation_cache_key, write_source
from ninetoothed.compiler.driver import Compilation
from ninetoothed.compiler.layout_runtime import memory_spans_overlap
from ninetoothed.compiler.materialization import dense_scratch_spec
from ninetoothed.ir import IndexExpr, LaunchABI, LaunchBinding, LaunchPlan, ir_to_dict


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
    """Emit a proved partition with ordinary tensor and stride launch bindings."""
    plan = compilation.artifact.execution_plans.get("materialization")

    if plan is None:
        return None

    originals = {spec.name: spec for spec in compilation.kernel.tensors}
    scratch = {tensor.name: dense_scratch_spec(tensor) for tensor in plan.scratch}

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
        from ninetoothed.compiler.passes import lower_for_target

        kernel = replace(
            kernel,
            ssa=lower_for_target(
                stage.program,
                backend=compilation.target.backend,
                target_context=compilation.target,
                compiler_options=options,
                kernel_metadata=kernel.metadata,
                tensors=specs,
                pass_pipeline=compilation.request.pipeline,
                pass_options=pass_options,
            ),
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


def materialize_staged(compilation, *, stage_materializer, output_dir=None):
    """Compile stages once and retain each call's scratch through async launches."""
    from ninetoothed.compiler.runtime import (
        Handle,
        _filter_runtime_kwargs,
        _runtime_wrapper,
        _verified_runtime_launch,
    )

    built = stage_compilations(compilation)

    if built is None:
        return None

    if compilation.artifact.backend.value != "cuda":
        raise ValueError("The SSA stage executor currently requires CUDA.")

    plan, stages = built
    handles = tuple(
        stage_materializer(
            stage,
            output_dir=None
            if output_dir is None
            else Path(output_dir) / f"stage-{index}",
        )
        for index, stage in enumerate(stages)
    )
    reference = replace(
        compilation,
        artifact=replace(
            compilation.artifact,
            execution_plans={
                name: value
                for name, value in compilation.artifact.execution_plans.items()
                if name != "materialization"
            },
        ),
    )
    fallback = stage_materializer(
        reference,
        output_dir=None if output_dir is None else Path(output_dir) / "fallback",
    )
    outputs = set(compilation.launch_abi.outputs)
    reads = {
        value.name for stage in plan.stages for value in stage.program.inputs
    } - outputs
    public = compilation.launch_abi.public_args
    device_argument = next(name for name in public if name in reads)
    artifact = replace(
        compilation.artifact,
        metadata={
            **compilation.artifact.metadata,
            "execution_route": "cuda-guarded-staged-ssa",
            "primary_source_role": "whole-call-alias-fallback",
            "materialization_plan": plan.metadata(),
            "materialized_stages": tuple(
                {
                    "kernel_name": stage.artifact.kernel_name,
                    "source_sha256": hashlib.sha256(
                        stage.artifact.primary_source.encode()
                    ).hexdigest(),
                    "ssa": stage.artifact.metadata["ssa"],
                    "launch_abi": ir_to_dict(stage.launch_abi),
                }
                for stage in stages
            ),
        },
    )
    compilation = replace(compilation, artifact=artifact)

    def aliases(arguments):
        # Storage ranges also cover overlapping DLPack imports with distinct bases.
        # Keep the original executor for every possibly aliased public output.
        def span(name):
            value = arguments[name]
            storage = value.untyped_storage()
            start = storage.data_ptr()

            return value.device, start, start + storage.nbytes()

        written = tuple(span(name) for name in outputs)

        return any(
            memory_spans_overlap(span(name), output)
            for name in public
            if name in reads
            for output in written
        )

    def invoke(*args, **kwargs):
        import torch

        arguments = dict(zip(public, args))
        arguments.update(kwargs)

        if aliases(arguments):
            return fallback(*args, **kwargs)

        device = arguments[device_argument].device

        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)

            for tensor in plan.scratch:
                arguments[tensor.name] = torch.empty(
                    tuple(map(int, tensor.shape)),
                    dtype=getattr(torch, tensor.dtype),
                    device=device,
                )

            launch_error = None

            try:
                for stage, handle in zip(stages, handles):
                    meta = _filter_runtime_kwargs(stage.launch_abi, kwargs)
                    meta = {
                        name: value
                        for name, value in meta.items()
                        if name not in stage.launch_abi.public_args
                    }
                    handle(
                        *(arguments[name] for name in stage.launch_abi.public_args),
                        **meta,
                    )
            except BaseException as error:
                launch_error = error
                raise
            finally:
                # A later stage may fail after earlier work has been queued.
                # Preserve the original launch error while protecting all owners.
                record_error = None

                tensor_names = (
                    spec.name for spec in compilation.kernel.tensors if spec.ndim > 0
                )

                for name in (*tensor_names, *(tensor.name for tensor in plan.scratch)):
                    try:
                        arguments[name].record_stream(stream)
                    except BaseException as error:
                        if record_error is None:
                            record_error = error

                if record_error is not None:
                    try:
                        stream.synchronize()
                    except BaseException:
                        if launch_error is None:
                            raise

                    if launch_error is None:
                        raise record_error

    launch = _verified_runtime_launch(
        _runtime_wrapper(
            invoke,
            compilation.launch_abi,
            low_level=False,
            specs=compilation.kernel.tensors,
            device_types=("cuda",),
        )
    )
    source = write_source(
        artifact.kernel_name,
        artifact.primary_source,
        "cu",
        cache_key=compilation_cache_key(compilation),
    )
    handle = Handle(compilation, handles, launch, source)
    handle._stage_handles = handles
    handle._stage_compilations = stages
    handle._materialization_plan = plan
    handle.stage_artifacts = tuple(stage.artifact for stage in stages)
    handle.stage_kernels = tuple(stage.kernel for stage in stages)

    def execution_artifacts_for(*args, **kwargs):
        arguments = dict(zip(public, args))
        arguments.update(kwargs)

        return (fallback._artifact,) if aliases(arguments) else handle.stage_artifacts

    handle.execution_artifacts_for = execution_artifacts_for

    return handle
