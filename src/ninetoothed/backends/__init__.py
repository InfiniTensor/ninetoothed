"""Public backend utilities."""

from dataclasses import replace

from ninetoothed.backends.core import (
    Artifact,
    Backend,
    BuiltArtifact,
    Capability,
    Registry,
    Target,
    normalize_target,
)
from ninetoothed.ir import Kernel


def create_default_registry() -> Registry:
    from ninetoothed.backends.cuda import CudaBackend
    from ninetoothed.backends.tilelang import TileLangBackend
    from ninetoothed.backends.triton import TritonBackend

    registry = Registry()
    registry.register(TritonBackend())
    registry.register(TileLangBackend())
    registry.register(CudaBackend())

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
) -> Artifact:
    target = normalize_target(backend)
    backend_impl = default_registry().get(target)
    backend_options = backend_impl.normalize_options(
        dict(kernel.compiler_options.get("backend_options", {}))
    )

    if backend_options != kernel.compiler_options.get("backend_options", {}):
        kernel = replace(
            kernel,
            compiler_options=dict(kernel.compiler_options)
            | {"backend_options": backend_options},
        )

    kernel = _prepare_kernel_for_backend(kernel, target)
    kernel = backend_impl.prepare_for_emission(kernel)

    return backend_impl.emit(kernel)


def _prepare_kernel_for_backend(kernel: Kernel, target: Target) -> Kernel:
    from ninetoothed.compiler.passes import lower_for_target

    ssa = kernel.ssa

    if ssa is None:
        return kernel

    if ssa.metadata.get("target_backend") != target.value:
        ssa = lower_for_target(
            ssa,
            backend=target,
            compiler_options=kernel.compiler_options,
            kernel_metadata=kernel.metadata,
            tensors=kernel.tensors,
            pass_pipeline=kernel.compiler_options.get("ssa_pass_pipeline"),
            pass_options=kernel.compiler_options.get("ssa_pass_options"),
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


def backend_capabilities() -> tuple[Capability, ...]:
    return default_registry().capabilities()


__all__ = [
    "Backend",
    "BuiltArtifact",
    "Artifact",
    "Capability",
    "Target",
    "Registry",
    "Kernel",
    "backend_capabilities",
    "create_default_registry",
    "default_registry",
    "emit",
    "normalize_target",
]
