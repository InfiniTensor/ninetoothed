# Entropy and Garbage Collection

Agents copy patterns from the repository, including weak patterns. This file is
the place to encode cleanup rules as they become known.

## Golden Rules

- Prefer current repository patterns over invented abstractions.
- Keep diffs local to the operator, tests, examples, and benchmark.
- Add regression tests before broad cleanup.
- Mark debt explicitly instead of hiding it in prose.
- Convert repeated review comments into linter checks.

## Drift Scan Checklist

- stale references:
  - Run `rg -n "references/|examples/|subagent-sessions/|scripts/|tests/|${NINETOOTHED_REPO}" SKILL.md references examples`.
  - For upstream NineToothed anchors, verify the path still exists under
    `${NINETOOTHED_REPO}` before treating the reference as current.
  - If a path is stale, either update it to the closest current upstream anchor
    or record the stale path in the debt ledger with the next verification step.
- duplicated templates:
  - Run `find subagent-sessions -maxdepth 2 -type f | sort`.
  - Keep `_template/` as the only reusable template source; generated dry-run
    sessions should be removed unless intentionally preserved as evidence.
  - If two templates describe the same fields differently, merge the newer
    contract into `_template/` and remove the duplicate.
- TODOs without owner:
  - Run `rg -n "TODO\b|FIXME|TBD" . --glob '!scripts/lint_skill_structure.py'`.
  - Every retained item must use `TODO(owner): action` or be moved into this
    ledger.
  - The structure lint now rejects unowned TODO markers.
- examples that no longer run:
  - Run the correctness command listed in each `examples/*/TASK.md` only when
    the upstream checkout and device/runtime are available.
  - If a command is not run, keep the explicit `Not run yet` placeholder rather
    than fabricating a result.
  - If a command fails, record the command, first failure signal, blocker or
    root cause, and rerun status in the example or failure playbook.
- scripts without tests:
  - Run `python scripts/lint_skill_structure.py .` and
    `python tests/test_structure.py` from the skill root, or use the full
    `skills/competition/ninetooth-operator-dev/...` path from the repository
    root.
  - New lint behavior needs a focused regression in `tests/test_structure.py`.
  - Helper scripts without executable checks belong in the debt ledger.
- benchmark claims without command:
  - Run `rg -n "faster|slower|speedup|slowdown|parity|regression" examples`.
  - Performance conclusions require a benchmark command plus real output or a
    concrete blocker under `## Benchmark Result`.
  - The structure lint rejects example performance claims while benchmark
    evidence is still missing or pending.
- unsupported scope not disclosed:
  - Run `rg -n "Unsupported Scope|unsupported scope" examples references`.
  - Each task example must name excluded dtype, shape, layout, device,
    benchmark, AOT, or autograd behavior when relevant.
  - If unsupported scope is missing, add it to the task contract or example
    before expanding implementation guidance.

## I10 Scan Snapshot

Date: 2026-05-21.

- Upstream path references: checked the referenced NineToothed anchors used by
  the repository map, pattern index, examples, and failure playbook; no missing
  upstream files were found in `${NINETOOTHED_REPO}`.
- Subagent sessions: only `_template/` files are present; no dry-run or stale
  generated session directories need cleanup.
- Examples: all four required `TASK.md` files exist and keep honest
  `Not run yet` placeholders for correctness or benchmark results that have not
  been executed.
- TODO scan: no unowned project TODO/FIXME/TBD markers outside the lint rule
  implementation itself.
- Benchmark claim scan: performance wording in examples is currently guardrail
  text, not a measured speedup or slowdown claim.
- Script coverage: `lint_skill_structure.py` is covered by
  `tests/test_structure.py`, including the I9 regression checks for unowned
  TODOs and benchmark-claim evidence.

## Debt Ledger

| Date | Area | Debt | Impact | Next action |
| --- | --- | --- | --- | --- |
| 2026-05-21 | examples | Correctness and benchmark result fields are intentionally still `Not run yet` in the four self-test task specs. | Final submission cannot claim executed self-test or performance evidence until commands are actually run or blockers are recorded. | In I11 or a dedicated execution pass, run each listed correctness command in `${NINETOOTHED_REPO}`; run benchmark commands where required, or record concrete device/runtime blockers. |
| 2026-05-21 | examples | Layout-sensitive benchmark is optional and out of scope unless requested, while other examples require benchmark or blocker evidence. | Future agents may confuse optional benchmark scope with missing required evidence. | Preserve the explicit optional wording; if performance is requested later, record shape, stride, storage offset, dtype, device, baseline, and real timing/blocker. |
| 2026-05-21 | subagent-sessions | No real subagent session records exist yet beyond the template. | The mechanism is structurally ready but lacks a worked example for future agents to imitate. | When the first log-heavy operator, benchmark, or failure task is delegated, scaffold a real session and keep only L0/L1 in the parent-facing return. |
| 2026-05-21 | final packaging | `HONOR_CODE.md` and `REFERENCE.md` are not part of the current skill package yet. | Final competition handoff remains incomplete even though the skill framework is coherent. | Create final handoff materials in I11 and verify no local PDF or converted rules Markdown is packaged. |
