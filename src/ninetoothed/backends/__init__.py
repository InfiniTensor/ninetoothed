"""Public backend utilities."""

from ninetoothed.backends.core import (
    Artifact,
    Backend,
    BackendOptions,
    BuiltArtifact,
    Capability,
    Options,
    Registry,
    Target,
    normalize_options,
    normalize_target,
)
from ninetoothed.ir import Kernel


def create_default_registry() -> Registry:
    from ninetoothed.backends.cuda import CudaBackend
    from ninetoothed.backends.tilelang import TileLangBackend
    from ninetoothed.backends.triton import TritonBackend
    from ninetoothed.backends.tvm import TvmBackend

    registry = Registry()
    registry.register(TritonBackend())
    registry.register(TileLangBackend())
    registry.register(CudaBackend())
    registry.register(TvmBackend())

    return registry


_DEFAULT_BACKENDS: Registry | None = None


def default_registry() -> Registry:
    global _DEFAULT_BACKENDS
    if _DEFAULT_BACKENDS is None:
        _DEFAULT_BACKENDS = create_default_registry()
    return _DEFAULT_BACKENDS


def emit(
    kernel: Kernel,
    backend: Target | str | None = None,
    options: Options | None = None,
) -> Artifact:
    if options is None:
        options = normalize_options(backend)

    kernel = _prepare_kernel_for_backend(kernel, options)
    backend_impl = default_registry().get(options.target)
    artifact = backend_impl.emit(kernel, options)

    return _attach_pipeline_metadata(artifact, kernel)


def _prepare_kernel_for_backend(kernel: Kernel, options: Options) -> Kernel:
    from ninetoothed.compiler.passes import lower_for_target

    ssa = kernel.ssa

    if ssa is None:
        return kernel

    if ssa.metadata.get("target_backend") != options.target.value:
        ssa = lower_for_target(
            ssa,
            backend=options.target,
            compiler_options=kernel.compiler_options,
            kernel_metadata=kernel.metadata,
            pass_pipeline=kernel.compiler_options.get("ssa_pass_pipeline"),
            pass_options=kernel.compiler_options.get("ssa_pass_options"),
            autotune=kernel.compiler_options.get("ssa_tuning", False),
        )

    return type(kernel)(
        kernel_name=kernel.kernel_name,
        source=kernel.source,
        source_path=kernel.source_path,
        source_language=kernel.source_language,
        entrypoint=kernel.entrypoint,
        launch_abi=kernel.launch_abi,
        launch_plan=kernel.launch_plan,
        tensors=kernel.tensors,
        compiler_options=kernel.compiler_options,
        metadata=dict(kernel.metadata)
        | {
            "ssa_pipeline": tuple(ssa.metadata.get("pass_trace", ())),
            "ssa_target_backend": ssa.metadata.get("target_backend"),
        },
        ssa=ssa,
    )


def _attach_pipeline_metadata(artifact: Artifact, kernel: Kernel) -> Artifact:
    if kernel.ssa is None:
        return artifact
    return Artifact(
        backend=artifact.backend,
        kernel_name=artifact.kernel_name,
        language=artifact.language,
        sources=artifact.sources,
        entrypoint=artifact.entrypoint,
        stage=artifact.stage,
        materializable=artifact.materializable,
        metadata=dict(artifact.metadata)
        | {
            "kernel_metadata": dict(kernel.metadata),
            "ssa_metadata": dict(kernel.ssa.metadata),
            "ssa_pass_trace": tuple(kernel.ssa.metadata.get("pass_trace", ())),
            "ssa_schedule": dict(kernel.ssa.metadata.get("schedule", {})),
            "ssa_pipeline_selection": dict(
                kernel.ssa.metadata.get("pipeline_selection", {})
            ),
            "ssa_optimization": dict(kernel.ssa.metadata.get("optimization", {})),
        },
    )


def backend_capabilities() -> tuple[Capability, ...]:
    return default_registry().capabilities()


__all__ = [
    "Backend",
    "BackendOptions",
    "BuiltArtifact",
    "Artifact",
    "Capability",
    "Target",
    "Options",
    "Registry",
    "Kernel",
    "backend_capabilities",
    "create_default_registry",
    "default_registry",
    "emit",
    "normalize_target",
    "normalize_options",
]
