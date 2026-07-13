# Final Report: NineToothed Operator Development Skill

## Submission Summary

- Track: T3-1-1, NineToothed operator development skill.
- Skill name: `ninetooth-operator-dev`.
- Skill package path: `skills/competition/ninetooth-operator-dev/`.
- Upstream NineToothed checkout: set by the user with `NINETOOTHED_REPO`.
- Upstream commit inspected during this iteration:
  `c9ebd4950a185beed8d4c1db9ff4a1fd133934ae`.

This repository now contains a progressive, offline `.skill` framework for AI
agents working on NineToothed operator tasks. The skill is organized as a thin
entry point, task-specific references, mechanical structure checks, self-test
task specifications, and subagent session templates.

## Package Contents

- `SKILL.md`: compact task entry point and hard rules.
- `agents/openai.yaml`: Codex/OpenAI interface metadata.
- `references/`: repository map, operator contract, DSL pattern index,
  verification matrix, performance diagnostics, failure playbook, subagent
  orchestration, entropy cleanup, and script index.
- `scripts/`: offline helper scripts for structure linting, upstream map
  collection, self-test scaffolding, and subagent session scaffolding.
- `examples/`: four self-test task specifications covering elementwise
  broadcast, reduction/block, layout-sensitive, and performance/AOT diagnostic
  work.
- `tests/`: structure regression tests and evaluation-case metadata.
- `subagent-sessions/_template/`: result-oriented subagent session template.

## Self-Test Materials

Four self-test task specifications are present:

1. `examples/elementwise-broadcast/TASK.md`: `bias_relu`.
2. `examples/reduction-block/TASK.md`: `row_softmax_tail`.
3. `examples/layout-sensitive/TASK.md`: `strided_affine_copy`.
4. `examples/performance-diagnostics/TASK.md`: `aot_add_autotune_diagnostics`.

Each task includes the input prompt, upstream anchors, expected patch surface,
correctness command, correctness result field, benchmark command or benchmark
scope, benchmark result field, and unsupported-scope notes.

The example commands have not been executed as final benchmark evidence in this
repository. The task files intentionally preserve `Not run yet` or optional
benchmark language where evidence has not been collected.

## Benchmark Materials

Benchmark command templates are included for:

- elementwise broadcast `bias_relu`;
- reduction/block `row_softmax_tail`;
- optional layout-sensitive `strided_affine_copy`;
- performance/AOT diagnostic `aot_add_autotune_diagnostics`.

No speedup, slowdown, parity, or regression claim is made without real timing
or blocker evidence. The structure lint rejects example performance conclusions
that claim a result while benchmark evidence is missing or pending.

## Mechanical Verification

Current verification entry points:

```bash
python skills/competition/ninetooth-operator-dev/scripts/lint_skill_structure.py skills/competition/ninetooth-operator-dev
python skills/competition/ninetooth-operator-dev/tests/test_structure.py
```

The structure checks validate required package files, example task headings,
subagent template headings, script executability, common secret signatures,
forbidden packaged artifacts, unowned debt markers, and unsupported performance
claims without benchmark evidence.

## Validation Environment Blocker

The packaging and structure tests run locally, but the upstream GPU suite cannot
run on the available macOS environment:

- platform: macOS (`darwin`);
- Python: 3.11.6;
- PyTorch: 2.10.0;
- CUDA available: `False`;
- Triton installation command: `python -m pip install "triton>=3.0.0"`;
- Triton installation result: `No matching distribution found for triton>=3.0.0`;
- full-suite command: `pytest`;
- full-suite result: `23 errors during collection`, all rooted in
  `ModuleNotFoundError: No module named 'triton'`.

The skill-specific pytest suite completed with `16 passed, 16 subtests passed`.
The four example operator tasks remain executable specifications with `Not run
yet` fields. The original proposal targets are not achieved results, and this
report makes no GPU correctness or benchmark claim.

## Compliance Notes

- The skill package does not include the local competition PDF or converted
  regulation Markdown.
- The repository `.gitignore` excludes those local rule artifacts.
- No API keys, account credentials, hidden evaluation answers, private data, or
  test-bypass instructions are intentionally included.
- The skill is designed for offline use against a local NineToothed checkout.
- AI assistance was used to draft, organize, and mechanically validate these
  materials; see `REFERENCE.md` for disclosure.

## Remaining Risks

- The four example tasks are executable specifications, not completed upstream
  patches with recorded pytest output.
- Final benchmark results still require a suitable local GPU/Triton/NineToothed
  runtime and explicit command execution.
- The first real delegated operator, benchmark, or failure task should create a
  concrete `subagent-sessions/` record from the provided template.
