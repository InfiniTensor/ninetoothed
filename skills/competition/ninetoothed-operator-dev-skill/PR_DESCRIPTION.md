# T3-1-1 NineToothed Operator Development Skill

## Submission Identity

- Skill: `ninetoothed-operator-dev-skill`
- Problem: `T3-1-1`
- Team: `123123`
- Participant: `刘李宏`
- GitHub ID: `Dreamt-Deer-Waking-Fish`

Honor Code identity and signature are completed in `HONOR_CODE.md`.

## Scope

This package guides an AI coding agent through repository discovery,
requirement extraction, minimal NineToothed/ntops implementation, PyTorch-based
correctness, layout analysis, correctness-gated benchmark work, generated
source/AOT diagnosis, InfiniCore dispatch checks, failure closure, and clean
patch handoff.

It does not cover unrelated repositories, ordinary Python tasks, compiler-core
redesign by default, or unverified hardware claims.

## Install And Use

Place the directory at:

```text
skills/competition/ninetoothed-operator-dev-skill/
```

Invoke `$ninetoothed-operator-dev-skill` for a NineToothed or ntops operator
task. Read `SKILL.md`, then load only the reference guide needed for the current
operator family.

## Self-Test Evidence

| Case | Main evidence | Status boundary |
| --- | --- | --- |
| EW | add/relu correctness and add raw timing CSV | Selected public CUDA checks only |
| RED | softmax server correctness and raw timing CSV | One contiguous float32 shape |
| LAYOUT | pooling baseline production patch and server correctness | Repaired test-only patch: 80 passed, 72 skipped |
| PERF/AOT | SiLU fallback patch and native-import diagnosis | Stub branches pass; native dispatch/AOT/timing blocked |
| IMPL | production wrappers, tests, clean applicable patch | Baseline-created artifact, not attributed to skill guidance |

## No-Skill Comparison

The comparison is mixed and intentionally preserves baseline strengths:

- Baseline produced the strongest real production change and layout server
  execution.
- Skill guidance improved requirement extraction, evidence status, failure
  records, and the reduction task's executable handoff.
- The skill-guided layout and performance-integration copies initially failed
  server apply-check; later local LF remediation is reported separately.

## Benchmarks

- add, `1024x1024`, float32 contiguous, warmup 10, repeat 30: PyTorch
  `0.017408 ms`, ntops `0.066000 ms`, baseline/candidate ratio `0.263758`.
- softmax, `64x1024`, float32 contiguous, warmup 10, repeat 30: PyTorch
  `0.023568 ms`, ntops `0.071728 ms`, baseline/candidate ratio `0.328575`.

Both records passed the benchmark script's correctness guard and retain all 30
timing samples. Both show selected-shape ntops regressions; they are short runs
and do not establish a broad performance conclusion.

## Validation

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -p "test_*.py" -v
PYTHONPYCACHEPREFIX="$(mktemp -d)" python -m compileall -q scripts tests
python scripts/validate_skill_package.py .
python scripts/check_no_secrets.py .
python scripts/check_false_verified_claims.py .
python scripts/check_markdown_links.py .
```

The final submission includes machine-readable validation logs and repeats the
same checks after clean ZIP extraction.

Recorded result:

- Current `skill-creator` quick validation: PASS.
- Self-contained unit tests: `20` passed.
- Python compilation: PASS.
- Package, secret, unsupported-claim, and Markdown-link validators: PASS.
- All nine script entry points returned successful `--help` output.
- Clean-copy and clean-ZIP-extraction validation: PASS.
- Ruff: SKIP because it is not installed in the project virtual environment.

## Compliance And Report

- Honor Code: `HONOR_CODE.md`
- References and AI disclosure: `REFERENCE.md`
- Report source: `reports/final_report.md`
- Report PDF: `reports/123123_九齿skill创新挑战_T3-1-1_赛题报告.pdf`
- Claim ledger: `reports/claim_ledger.md`
- Submission checklist: `SUBMISSION_CHECKLIST.md`
- Internal file hashes: `MANIFEST.sha256`
- Final archive: `ninetoothed-operator-dev-skill_T3-1-1_final_signed_20260712.zip`

## Known Limits

- No hidden evaluation result is claimed.
- Full non-contiguous support is not claimed.
- Generated-source runtime output, AOT output, and InfiniCore dispatch are not
  established.
- Fork URL, branch push, pull request creation, and platform upload remain manual participant actions.


## Final Release Target

- Fork URL: participant must use the confirmed personal fork before pushing.
- Suggested branch: `2026-spring-Dreamt-Deer-Waking-Fish-T3-1-1`
- Intended submission directory: `skills/competition/ninetoothed-operator-dev-skill/`
