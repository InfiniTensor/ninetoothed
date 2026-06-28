"""TileLang backend implementation.

TileLang lowering is driven by the same SSA operation stream as Triton, CUDA,
and TVM.  This file intentionally contains no kernel-specialized emitters.
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


class TileLangBackend(Backend):
    name = BackendName.TILELANG
    capability = BackendCapability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Unified SSA backend; TileLang source is emitted from SSA operations.",
            "Backend-specific behavior is expressed as operation-level rendering only.",
        ),
    )

    def lower(
        self, kernel: KernelIR, options: BackendOptions | None = None
    ) -> BackendArtifact:
        return lower_unified_ssa_artifact(kernel, self.name)
