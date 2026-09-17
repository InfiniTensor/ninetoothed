# References And Disclosure

## Contest Materials

The package was designed against the locally preserved competition documents:

- `九齿 .skill 创新挑战赛道赛题与规则.md`, version 0.8.
- `2026春季启元人工智能大赛指南.md`.
- `2026 春季启元人工智能大赛赛题.md`.

They define T3-1-1, the four self-test families, two required benchmark cases,
report content, Honor Code, Reference disclosure, and suggested submission
layout. These documents are cited by title and are not copied into the skill.

## Source Repositories

### ntops

- Repository: <https://github.com/InfiniTensor/ntops>
- Recorded revision: `6bc90d5aba29146a8757fe0b67e7e1966b92bb5f`
- License: Apache License 2.0, verified from the local `LICENSE` file.
- Use: operator patterns, wrappers, tests, and target source for elementwise,
  reduction, layout, benchmark, and patch evidence.

### InfiniCore

- Repository: <https://github.com/InfiniTensor/InfiniCore>
- Recorded revision: `d2758a5c3b28c70edb3743ca8cdb5cdbd97d237c`
- License: MIT License, verified from the local `LICENSE` file.
- Use: generated-source, AOT, Python dispatch, native import, and integration
  paths; target source for the SiLU fallback patch evidence.

The submission archive does not vendor either repository. Included patch files
are compact evidence against the recorded revisions.

## Dependencies

- Python 3.10 or newer: package validators use only the standard library.
- Git: revision discovery and patch applicability checks.
- ntops 0.1.0: local metadata lists `ninetoothed>=0.16.0` and `torch`.
- PyTorch: correctness reference and CUDA timing when available.
- Triton and NineToothed: required by ntops runtime paths.
- InfiniCore native package: required only for actual dispatch verification.

No new online dependency was downloaded during finalization. Runtime tools are
optional until a task requires their corresponding check; unavailable device
work must remain `[TODO-GPU]` or `[BLOCKED]`.

## Reused And Adapted Project Material

The following scripts were adapted from the project's existing development
package and retained because they implement deterministic, reusable checks:

- `scripts/run_ntops_microbenchmark.py`
- `scripts/probe_infinicore_dispatch.py`
- `scripts/collect_source_tree_manifest.py`
- `scripts/validate_patch_artifact.py`

Final package validators, self-contained tests, references, report, and
submission files were rewritten for this distribution. Raw timing CSV files
and compact server-output excerpts are derived from recorded project evidence;
they are not synthetic results.

## AI Assistance Disclosure

OpenAI Codex was used to inspect local materials, normalize patch line endings,
author and edit the skill instructions and references, create validation tools
and tests, assemble evidence, and generate the report and archive. AI-generated
text and code must be reviewed by the participant before submission. Actual
test counts, revisions, patch outcomes, and benchmark timings are tied to
preserved command output or artifacts and are not inferred from AI prose.

## Evidence Boundaries

- No hidden evaluation result is available or claimed.
- Full non-contiguous support is not claimed.
- Generated-source runtime output, AOT output, and InfiniCore dispatch remain
  unverified because the recorded native import was blocked.
- The add and softmax timings are short, selected-shape, single-GPU records and
  do not establish broad speedup.
- Historical server apply failures remain disclosed even where later local
  normalization produced a clean apply-check.

## Participant Review

The participant must complete the identity/signature block in `HONOR_CODE.md`,
review these disclosures, and confirm that no additional external material was
added before upload.
