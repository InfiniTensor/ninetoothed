# Final Submission Guide

Use this reference when preparing competition artifacts.

## Required Files

The final package should include:

- `SKILL.md`
- `references/`
- `scripts/`
- `examples/`
- `tests/` or self-test notes
- `README.md`
- `HONOR_CODE.md`
- `REFERENCE.md`
- final report PDF or report source

## Repository Contribution Rules (enforced by hooks and CI)

The `ninetoothed` repository enforces these via `.githooks` and the `contributing.yml` CI workflow (`scripts/check_contributing_metadata.py`):

- Branch name: lowercase kebab-case, at most 50 characters, e.g. `spring-2026-<githubid>-t3-1-1`.
- Commit message and PR title: capitalized first letter, imperative mood, no trailing punctuation.
- PR description must include pytest output.
- Run before pushing: `python scripts/check_contributing_style.py --fix`, `ruff format`, `ruff check`, `python scripts/check_contributing_style.py`, `pytest`.
- Enable hooks after cloning: `git config core.hooksPath .githooks`.

## PR Description Checklist

Include:

- skill name
- problem id: T3-1-1
- team name
- applicable scope
- unsupported scope
- installation and usage
- self-test task records
- before/after comparison with and without the skill
- links or attachments for proposal and final report
- `HONOR_CODE.md` and `REFERENCE.md`

## Final Report Checklist

Include:

- skill goal and design principles
- package structure
- core workflow
- four self-test operator tasks
- correctness commands and results
- at least two benchmark designs and results
- failure diagnosis case
- comparison against an agent without this skill
- safety, dependency, license, and citation disclosure
- maintenance plan

## Award Threshold Focus

Optimize for:

- at least 5 of 8 hidden tasks with task-completion score >= 3/4
- hidden-task raw score >= 48/80
- at least 2 hidden tasks with real performance evidence
- no safety or reproducibility violation

## Compliance Rules

- No secrets.
- No hidden answers.
- No online-only dependency.
- No test bypass.
- No undisclosed copied code.
- No broad unrelated refactor.
