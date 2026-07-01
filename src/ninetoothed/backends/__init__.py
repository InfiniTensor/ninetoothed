"""Public backend utilities."""

from __future__ import annotations

from ninetoothed.backends.base import (
    Backend,
    BackendArtifact,
    BackendCapability,
    BackendName,
    BackendOptions,
    BackendRegistry,
    normalize_backend_name,
    normalize_backend_options,
)
from ninetoothed.backends.cuda import CudaBackend
from ninetoothed.backends.tilelang import TileLangBackend
from ninetoothed.backends.triton import TritonBackend
from ninetoothed.backends.tvm import TvmBackend
from ninetoothed.ir import KernelIR


def create_default_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(TritonBackend())
    registry.register(TileLangBackend())
    registry.register(CudaBackend())
    registry.register(TvmBackend())

    return registry


DEFAULT_BACKENDS = create_default_registry()


def lower(
    kernel: KernelIR,
    backend: BackendName | str | None = None,
    options: BackendOptions | None = None,
) -> BackendArtifact:
    if options is None:
        options = normalize_backend_options(backend)

    kernel = _prepare_kernel_for_backend(kernel, options)
    backend_impl = DEFAULT_BACKENDS.get(options.name)
    artifact = backend_impl.lower(kernel, options)

    return _attach_pipeline_metadata(artifact, kernel)


def _prepare_kernel_for_backend(kernel: KernelIR, options: BackendOptions) -> KernelIR:
    from ninetoothed.ssa_passes import lower_ssa_for_backend

    ssa = kernel.ssa

    if ssa is None:
        return kernel

    if ssa.metadata.get("target_backend") != options.name.value:
        ssa = lower_ssa_for_backend(
            ssa,
            backend=options.name,
            compiler_options=kernel.compiler_options,
            kernel_metadata=kernel.metadata,
        )

    return type(kernel)(
        kernel_name=kernel.kernel_name,
        source=kernel.source,
        source_path=kernel.source_path,
        source_language=kernel.source_language,
        entrypoint=kernel.entrypoint,
        launch=kernel.launch,
        tensors=kernel.tensors,
        compiler_options=kernel.compiler_options,
        metadata=dict(kernel.metadata)
        | {
            "ssa_pipeline": tuple(ssa.metadata.get("pass_trace", ())),
            "ssa_target_backend": ssa.metadata.get("target_backend"),
        },
        program=kernel.program,
        ssa=ssa,
    )


def _attach_pipeline_metadata(
    artifact: BackendArtifact, kernel: KernelIR
) -> BackendArtifact:
    if kernel.ssa is None:
        return artifact
    return BackendArtifact(
        backend=artifact.backend,
        kernel_name=artifact.kernel_name,
        language=artifact.language,
        sources=artifact.sources,
        entrypoint=artifact.entrypoint,
        executable=artifact.executable,
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


def backend_capabilities() -> tuple[BackendCapability, ...]:
    return DEFAULT_BACKENDS.capabilities()


__all__ = [
    "Backend",
    "BackendArtifact",
    "BackendCapability",
    "BackendName",
    "BackendOptions",
    "BackendRegistry",
    "CudaBackend",
    "DEFAULT_BACKENDS",
    "KernelIR",
    "TileLangBackend",
    "TritonBackend",
    "TvmBackend",
    "backend_capabilities",
    "create_default_registry",
    "lower",
    "normalize_backend_name",
    "normalize_backend_options",
]
