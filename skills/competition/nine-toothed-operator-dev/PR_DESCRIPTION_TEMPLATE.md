# [2026 Spring][T3-1-1] <GitHub ID>

## Skill

- Name: `nine-toothed-operator-dev`
- Team:
- Problem: T3-1-1 NineToothed Operator Development Skill

## Scope

Applicable:

- Elementwise and broadcast operators
- Reduction and block operators
- Non-contiguous, stride, and offset-sensitive tasks
- Correctness tests against PyTorch or existing implementations
- Benchmark, generated source, AOT build, and failing-test diagnosis

Not applicable:

- NineToothed compiler core redesign
- Hidden evaluation answers or task-specific bypasses
- Online-only services or private credentials

## Installation and Usage

Place the skill under:

```text
skills/competition/nine-toothed-operator-dev/
```

Prompt:

```text
Use the nine-toothed-operator-dev skill to implement this NineToothed operator task. Follow the required workflow, add correctness tests, run targeted validation, include benchmark evidence if performance-sensitive, and finish with changed files, commands, results, unsupported cases, and residual risk.
```

## Self-Test Records

| Task | Category | Correctness | Benchmark | Notes |
| --- | --- | --- | --- | --- |
| 01-elementwise-add | Elementwise/broadcast | TODO | TODO | TODO |
| 02-softmax-reduction | Reduction/block | TODO | TODO | TODO |
| 03-layout-stride-offset | Layout-sensitive | TODO | TODO/NA | TODO |
| 04-performance-diagnosis | Performance/diagnosis | TODO | TODO | TODO |

## Before/After Comparison

| Task | Without Skill | With Skill | Improvement |
| --- | --- | --- | --- |
| Elementwise | TODO | TODO | TODO |
| Reduction | TODO | TODO | TODO |
| Layout | TODO | TODO | TODO |
| Diagnosis | TODO | TODO | TODO |

## Validation Commands

```shell
python scripts/check_submission.py --skill-dir .
pytest <targeted-correctness-command>
pytest -m benchmark <targeted-benchmark-command>
```

## Compliance

- [ ] `HONOR_CODE.md` signed
- [ ] `REFERENCE.md` completed
- [ ] No secrets or credentials
- [ ] No hidden answers
- [ ] No test bypasses
- [ ] External references disclosed

## Attachments

- Proposal:
- Final report:
- Self-test logs:
