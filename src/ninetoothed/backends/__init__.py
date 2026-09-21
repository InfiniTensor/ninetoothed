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
from ninetoothed.targets import (
    TargetContext,
    resolve_target_context,
    target_backend_options,
)


def create_default_registry() -> Registry:
    from ninetoothed.backends.bangc import BangCBackend
    from ninetoothed.backends.cuda import CudaBackend
    from ninetoothed.backends.tilelang import TileLangBackend
    from ninetoothed.backends.triton import TritonBackend

    registry = Registry()
    registry.register(TritonBackend())
    registry.register(TileLangBackend())
    registry.register(CudaBackend())
    registry.register(BangCBackend())

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
    *,
    target_context: TargetContext | None = None,
) -> Artifact:
    target = normalize_target(backend)
    target_context = target_context or resolve_target_context(target)

    if target_context.backend != target:
        raise ValueError(
            f"Target context backend `{target_context.backend.value}` does not match "
            f"emitter backend `{target.value}`."
        )

    backend_impl = default_registry().get(target)
    backend_options = backend_impl.normalize_options(
        target_backend_options(
            target_context,
            dict(kernel.compiler_options.get("backend_options", {})),
        )
    )

    if backend_options != kernel.compiler_options.get("backend_options", {}):
        kernel = replace(
            kernel,
            compiler_options=dict(kernel.compiler_options)
            | {"backend_options": backend_options},
        )

    target_metadata = target_context.as_metadata()
    kernel = replace(
        kernel,
        compiler_options=dict(kernel.compiler_options) | {"target": target_metadata},
        metadata=dict(kernel.metadata) | {"target": target_metadata},
    )
    kernel = _prepare_kernel_for_backend(kernel, target, target_context)
    kernel = backend_impl.prepare_for_emission(kernel)

    return backend_impl.emit(kernel)


def _prepare_kernel_for_backend(
    kernel: Kernel,
    target: Target,
    target_context: TargetContext,
) -> Kernel:
    from ninetoothed.compiler.passes import lower_for_target, validate_for_target

    ssa = kernel.ssa

    if ssa is None:
        return kernel

    if ssa.metadata.get("target") != target_context.as_metadata():
        ssa = lower_for_target(
            ssa,
            backend=target,
            target_context=target_context,
            compiler_options=kernel.compiler_options,
            kernel_metadata=kernel.metadata,
            tensors=kernel.tensors,
            pass_pipeline=kernel.compiler_options.get("ssa_pass_pipeline"),
            pass_options=kernel.compiler_options.get("ssa_pass_options"),
        )

    ssa = validate_for_target(
        ssa,
        backend=target,
        target_context=target_context,
        compiler_options=kernel.compiler_options,
        kernel_metadata=kernel.metadata,
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
            "ssa_target_platform": ssa.metadata.get("target_platform"),
            "ssa_target_compute_arch": ssa.metadata.get("target_compute_arch"),
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
