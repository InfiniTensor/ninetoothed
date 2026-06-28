# NineToothed Compiler Design Notes

This directory contains the design documents that should accompany the compiler implementation changes. Keep generated benchmark logs, large backend artifacts, and exploratory reports out of the main repository history.

## Recommended Reading Order

1. `ir_design_zh.md`
   - Overall IR motivation, expressiveness, extensibility, and current scope.
2. `ssa_ir_lowering_pipeline_zh.md`
   - Source-to-SSA lowering layers and target-annotated SSA flow.
3. `ssa_pass_registry_pipeline_zh.md`
   - Pass registry, default pipeline, backend-specific optimization pass organization.
4. `multi_backend_ir.md`
   - Earlier multi-backend design context and compatibility notes.

## Repository Boundary

The main `ninetoothed` repository should keep:

- compiler source code under `src/`
- tests under `tests/`
- reusable validation scripts under `scripts/`
- concise design documents under `docs/design/`

Generated reports and backend artifacts should remain local validation output unless a specific report is intentionally selected for a release note or PR discussion.

AI/Codex workflow skills are maintained separately in the `ninetoothed.skill` repository.
