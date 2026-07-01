"""Fan out backend-specific SSA pass registrations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ninetoothed.backends.base import BackendName

if TYPE_CHECKING:
    from ninetoothed.ssa_passes import SSAPassRegistry


def register_backend_specific_ssa_passes(registry: "SSAPassRegistry") -> None:
    from ninetoothed.backends.cuda import register_ssa_passes as register_cuda
    from ninetoothed.backends.tilelang import register_ssa_passes as register_tilelang
    from ninetoothed.backends.triton import register_ssa_passes as register_triton
    from ninetoothed.backends.tvm import register_ssa_passes as register_tvm

    register_triton(registry)
    register_cuda(registry)
    register_tilelang(registry)
    register_tvm(registry)
    _validate_backend_pass_contracts(registry)


def _validate_backend_pass_contracts(registry: "SSAPassRegistry") -> None:
    missing: list[str] = []

    for backend in BackendName:
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


def _required_backend_pass_names(backend: BackendName) -> tuple[str, ...]:
    return (
        f"ssa.{backend.value}.optimize_schedule",
        f"ssa.{backend.value}.lower_memory_scopes",
        f"ssa.{backend.value}.lower_intrinsics",
    )
