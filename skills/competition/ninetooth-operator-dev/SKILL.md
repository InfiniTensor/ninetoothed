---
name: ninetooth-operator-dev
description: Use when an AI agent must develop, verify, optimize, debug, or integrate NineToothed operators, including arrangement/application design, Tensor metadata, correctness tests, generated source inspection, AOT build checks, benchmarks, and failure diagnosis.
---

# NineToothed Operator Development

This skill guides an AI agent through a complete NineToothed operator task while
keeping the repository as the source of truth. Start small, read only the maps
needed for the task, and leave an auditable verification trail.

## Quick Start

1. Restate the operator requirement: semantics, inputs, outputs, shape, dtype,
   broadcasting, layout, boundary cases, and unsupported scope.
2. Read the current repository before designing. Begin with
   [repo-map.md](references/repo-map.md).
3. Classify the task:
   - elementwise or broadcast operator;
   - reduction or blocked operator;
   - layout-sensitive operator;
   - performance, generated source, AOT, or integration diagnosis.
4. Decide whether the work must be delegated to a subagent. If the goal is
   clear, result-only, and likely to produce noisy tool logs, use
   [subagent-orchestration.md](references/subagent-orchestration.md).
5. Load the matching reference:
   - requirement extraction: [operator-task-contract.md](references/operator-task-contract.md)
   - DSL shape: [dsl-pattern-index.md](references/dsl-pattern-index.md)
   - testing and scoring: [verification-matrix.md](references/verification-matrix.md)
   - performance work: [performance-diagnostics.md](references/performance-diagnostics.md)
   - failures: [failure-playbook.md](references/failure-playbook.md)
   - subagents: [subagent-orchestration.md](references/subagent-orchestration.md)
   - drift cleanup: [entropy-gc.md](references/entropy-gc.md)
6. Make the smallest repository-style change that can pass the task.
7. Verify with correctness tests first. Add benchmark or generated-source/AOT
   evidence for performance-sensitive work.
8. Report changed files, commands, results, known risks, and unsupported cases.

## Mandatory Self-Test Workflow

For every self-test operator task, the agent must:

1. Extract input, output, shape, dtype, broadcast, boundary, layout, and
   unsupported-scope requirements before implementation.
2. Read relevant NineToothed arrangement, application, tensor meta-operation,
   load/store, generated-source/AOT, benchmark, test, and example anchors.
3. Choose a NineToothed DSL expression that matches nearby repository style.
4. Write a correctness test aligned with PyTorch or an existing repository
   oracle.
5. Inspect generated source, AOT output, or benchmark results whenever the task
   is benchmark-required or performance-sensitive.
6. Record optimization notes for performance-sensitive work, such as redundant
   load/store, avoidable broadcast work, layout/stride assumptions, tile/block
   choices, or AOT/benchmark follow-up.
7. Record failure symptoms, diagnosis path, root cause or blocker, minimal fix
   or workaround, rerun command, and result when anything fails.
8. Name unsupported dtype, dynamic shape, non-contiguous layout, hardware,
   benchmark, AOT, or generated-source scope explicitly.

## Required Task Trace

Every completed task should leave these fields in the agent response or example
log:

- task summary;
- files inspected;
- implementation decision;
- files changed;
- correctness command and result;
- benchmark or generated-source/AOT command and result, when applicable;
- failure diagnosis and minimal fix, when applicable;
- performance conclusion or explicit non-requirement/blocker;
- subagent session path and progressive summary, when delegated;
- unsupported scope and residual risk.

## Mechanical Checks

Before packaging or submitting this skill framework, run:

```bash
python skills/competition/ninetooth-operator-dev/scripts/lint_skill_structure.py skills/competition/ninetooth-operator-dev
python skills/competition/ninetooth-operator-dev/tests/test_structure.py
```

## Hard Rules

- Do not rely on information outside the repository unless it is copied or
  cited as a versioned artifact.
- Do not put large manuals in `SKILL.md`; add navigable reference files instead.
- Do not add secrets, hidden evaluation answers, private data, or online-only
  runtime dependencies.
- Do not remove tests, fake results, or bypass build/benchmark failures.
- Do not make broad refactors unless the task explicitly requires them.
