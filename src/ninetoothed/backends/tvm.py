"""TVM backend implementation.

TVMScript generation is handled by the unified SSA emitter.  The backend class
is deliberately small so that future optimizations enter through SSA passes,
not through kernel-specialized lowerer branches.
"""

from __future__ import annotations

from ninetoothed.backends.base import (
    Backend,
    BackendArtifact,
    BackendCapability,
    BackendName,
    BackendOptions,
)
from ninetoothed.backends.ssa_unified import lower_unified_ssa_artifact
from ninetoothed.ir import KernelIR


class TvmBackend(Backend):
    name = BackendName.TVM
    capability = BackendCapability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Unified SSA backend; TVMScript is emitted from fine-grained SSA.",
            "No operator-specific emitters are used by the backend entrypoint.",
        ),
    )

    def lower(
        self, kernel: KernelIR, options: BackendOptions | None = None
    ) -> BackendArtifact:
        return lower_unified_ssa_artifact(kernel, self.name)
