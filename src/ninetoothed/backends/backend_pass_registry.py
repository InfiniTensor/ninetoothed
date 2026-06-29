"""Fan out backend-specific SSA pass registrations."""

from __future__ import annotations

from ninetoothed.ssa_passes import SSAPassRegistry


def register_backend_specific_ssa_passes(registry: SSAPassRegistry) -> None:
    from ninetoothed.backends.cuda import register_ssa_passes as register_cuda
    from ninetoothed.backends.tilelang import register_ssa_passes as register_tilelang
    from ninetoothed.backends.triton import register_ssa_passes as register_triton
    from ninetoothed.backends.tvm import register_ssa_passes as register_tvm

    register_triton(registry)
    register_cuda(registry)
    register_tilelang(registry)
    register_tvm(registry)
