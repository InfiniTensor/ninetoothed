"""Fan out backend-specific SSA pass registrations."""

from typing import TYPE_CHECKING

from ninetoothed.backends.core import Target

if TYPE_CHECKING:
    from ninetoothed.compiler.passes import OptimizeSchedule, Registry


def register_pass_bundle(
    registry: "Registry",
    *,
    backend: Target,
    optimize_schedule: type["OptimizeSchedule"],
) -> None:
    """Register the schedule pass implemented by one backend."""
    registry.register(
        optimize_schedule,
        tags=("optimization", backend.value),
    )


def register_passes(registry: "Registry") -> None:
    from ninetoothed.backends.cuda import register_ssa_passes as register_cuda
    from ninetoothed.backends.tilelang import register_ssa_passes as register_tilelang
    from ninetoothed.backends.triton import register_ssa_passes as register_triton

    register_triton(registry)
    register_cuda(registry)
    register_tilelang(registry)
    _validate_backend_pass_contracts(registry)


def _validate_backend_pass_contracts(registry: "Registry") -> None:
    missing: list[str] = []

    for backend in Target:
        for name in _required_backend_pass_names(backend):
            try:
                descriptor = registry.get(name)
            except KeyError:
                missing.append(name)
                continue

            if not descriptor.supports(backend):
                missing.append(name)

    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Backend SSA pass contract is incomplete: {names}.")


def _required_backend_pass_names(backend: Target) -> tuple[str, ...]:
    return (f"ssa.{backend.value}.optimize_schedule",)
