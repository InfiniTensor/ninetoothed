# NineToothed Operator Development Skill

Competition: 2026 Spring AI Competition, NineToothed .skill Innovation Track

Problem: T3-1-1 NineToothed Operator Development Skill

## Goal

This skill guides an AI agent to implement, test, benchmark, diagnose, and integrate NineToothed operators with a minimal reproducible patch.

It covers:

- elementwise and broadcast operators
- reduction and block operators
- non-contiguous, stride, and offset-sensitive tasks
- correctness tests against PyTorch or existing trusted implementations
- generated source, AOT build, benchmark, and failure diagnosis tasks
- arrangement debugging with `simulate_arrangement` or concrete `arranged.eval()` checks

It does not intentionally cover:

- modifying the NineToothed compiler core
- online-only services
- hidden evaluation answers
- hardware-specific hacks that cannot be reproduced by the judges

## Package Structure

```text
nine-toothed-operator-dev/
  SKILL.md
  references/
  scripts/
  examples/
  tests/
  reports/
  README.md
  HONOR_CODE.md
  REFERENCE.md
```

## Installation

Copy this folder into the competition-designated skills directory. If no directory is specified, use the structure requested by the organizers, for example:

```text
skills/competition/nine-toothed-operator-dev/
```

Then start the AI agent in the NineToothed repository and ask it to use the skill for operator development tasks.

## Suggested Usage Prompt

```text
Use the nine-toothed-operator-dev skill to implement this NineToothed operator task. Follow the required workflow, add correctness tests, run targeted validation, include benchmark evidence if performance-sensitive, and finish with changed files, commands, results, unsupported cases, and residual risk.
```

## Self-Test Tasks

This package includes four self-test task records under `examples/`:

1. `01-elementwise-add`
2. `02-softmax-reduction`
3. `03-layout-stride-offset`
4. `04-performance-diagnosis`

The current self-test records already include the first round of RTX 4090 validation logs. Re-run and update them if the organizers provide a different final repository or evaluation environment.

## Reference Notes

The most important files for an AI agent are:

- `SKILL.md`: required workflow and hard rules.
- `references/nine-toothed-api-notes.md`: NineToothed symbolic tensor, arrangement, application, debugging, and AOT notes.
- `references/operator-patterns.md`: implementation patterns by operator family.
- `references/repository-pattern-index-ninetoothed.md`: public NineToothed repository file index.
- `references/repository-pattern-index-examples.md`: public NineToothed examples repository file index.

## Validation

Run:

```shell
python scripts/check_submission.py --skill-dir .
```

This checks file presence and self-test record structure. It does not replace running the real NineToothed tests or benchmarks.
