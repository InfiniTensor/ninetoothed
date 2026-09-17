# NineToothed Operator Development Skill

This offline-capable skill guides an AI coding agent from an operator request to
a minimal implementation, correctness evidence, performance-aware validation,
failure closure, and an applicable repository patch.

Chinese guide: [README.zh-CN.md](README.zh-CN.md)

## Scope

Use it for NineToothed and ntops kernels, wrappers, exports, tests,
elementwise broadcasting, reductions, layout-sensitive behavior, generated
source, AOT builds, benchmarks, regressions, and InfiniCore dispatch.

Do not use it for unrelated Python work, ordinary prose, other repositories, or
unverified hardware claims. It does not modify the NineToothed compiler core by
default.

## Package Layout

```text
ninetoothed-operator-dev-skill/
  SKILL.md
  agents/openai.yaml
  references/
  scripts/
  examples/selftests/
  tests/
  reports/
  HONOR_CODE.md
  REFERENCE.md
  PR_DESCRIPTION.md
  SUBMISSION_CHECKLIST.md
  SUBMISSION_COMMANDS.md
```

## Install And Use

Copy this directory into the skill location supported by the evaluation
environment, or invoke it directly as `$ninetoothed-operator-dev-skill`.
Read `SKILL.md` first. Open only the reference guide needed for the current
operator family.

The core loop is:

1. Discover the repository and revision.
2. Write a requirement card.
3. Find the closest kernel, wrapper, export, and test pattern.
4. Implement the smallest correct change.
5. Run focused correctness and close failures.
6. Benchmark only after correctness.
7. Validate the patch against the intended clean revision.

## Self-Tests

| Case | Coverage | Evidence boundary |
| --- | --- | --- |
| `SELFTEST-EW-001` | add, elementwise workflow | Real short GPU timing for one contiguous float32 shape. |
| `SELFTEST-RED-001` | softmax, reduction workflow | Real short GPU timing for one contiguous float32 shape. |
| `SELFTEST-LAYOUT-001` | max/avg pooling layouts | Baseline GPU correctness; guided patch later repaired locally. |
| `SELFTEST-PERF-AOT-001` | SiLU integration diagnosis | Patch applicability repaired locally; native dispatch remains blocked. |
| `SELFTEST-IMPL-001` | production wrapper plus tests | Real production-code patch, server correctness, and clean apply-check. |

The baseline produced the strongest production change in the layout case. This
advantage is retained rather than hidden. The skill-guided result was stronger
in process structure but initially weaker in server patch handoff.

## Recorded Benchmarks

| Operator | Shape | DType | Layout | Warmup | Repeat | Baseline median | Candidate median |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| add | `1024x1024` | float32 | contiguous | 10 | 30 | 0.017408 ms | 0.066000 ms |
| softmax | `64x1024` | float32 | contiguous | 10 | 30 | 0.023568 ms | 0.071728 ms |

These are correctness-gated, single-GPU, selected-shape short runs with all 30
samples preserved. Both show an ntops regression for the selected input, not a
broad performance conclusion.

## Validation

Run from this directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -p "test_*.py" -v
PYTHONPYCACHEPREFIX="$(mktemp -d)" python -m compileall -q scripts tests
python scripts/validate_skill_package.py .
python scripts/check_no_secrets.py .
python scripts/check_false_verified_claims.py .
python scripts/check_markdown_links.py .
```

The same commands should pass after extracting the submission archive into a
new directory. The external bytecode cache keeps validation-generated
`__pycache__` files out of the skill tree.

## Dependencies And Fallbacks

- Package validation uses only the Python standard library and Git.
- Runtime operator checks require the repository's own Python environment.
- CUDA benchmarks require PyTorch, Triton, NineToothed, ntops, and a compatible
  GPU. When unavailable, keep timing fields unset and mark the task
  `[TODO-GPU]`.
- InfiniCore dispatch checks require a working native `infinicore.lib` import.
  Stop dispatch and timing claims when that import is blocked.

## Known Limits

- No hidden evaluation result is claimed.
- Full non-contiguous support is not claimed.
- Generated-source runtime, AOT output, and InfiniCore dispatch are not
  established by the included server evidence.
- Participant identity and signature must be completed before submission; see
  `HONOR_CODE.md` and `SUBMISSION_CHECKLIST.md`.
