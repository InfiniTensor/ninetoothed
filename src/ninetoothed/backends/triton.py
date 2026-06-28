"""Triton backend implementation.

The backend lowerer is intentionally SSA-first.  It does not classify kernels
by operator family before emitting code; it delegates to the unified SSA
emitter, whose dispatch unit is a single SSA operation.
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


class TritonBackend(Backend):
    name = BackendName.TRITON
    capability = BackendCapability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Unified SSA backend; Triton source is generated from SSA operations.",
            "No source passthrough or kernel-specialized fallback is used.",
        ),
    )

    def lower(
        self, kernel: KernelIR, options: BackendOptions | None = None
    ) -> BackendArtifact:
        return lower_unified_ssa_artifact(kernel, self.name)
