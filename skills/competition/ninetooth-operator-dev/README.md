# NineToothed Operator Development Skill

English | [中文](README_zh.md)

## Project Overview

This directory contains an out-of-the-box AI-agent skill focused on operator
development workflows in the NineToothed ecosystem. The work is prepared for
the `.skill` innovation challenge, task T3-1-1, and is maintained in the
NineToothed repository under `skills/competition/ninetooth-operator-dev/`.

The current handoff includes the installable skill framework, four self-test
task specifications, mechanical validation scripts, final report, honor-code
statement, and reference disclosure. The self-test task files intentionally keep
unexecuted correctness or benchmark results marked as `Not run yet` where
evidence has not been collected.

## Quick Start

Use this directory as a skill package within a NineToothed checkout.

1. Clone NineToothed and enter the checkout:

```bash
git clone https://github.com/InfiniTensor/ninetoothed.git
cd ninetoothed
export NINETOOTHED_REPO="$PWD"
export NINETOOTHED_SKILL_DIR="$NINETOOTHED_REPO/skills/competition/ninetooth-operator-dev"
```

2. Install or expose the skill directory to your agent runtime. For Codex-style
   local skills, copy the skill directory into your user skill directory:

```bash
mkdir -p ~/.codex/skills
cp -R "$NINETOOTHED_SKILL_DIR" ~/.codex/skills/
```

If your agent runtime uses a different skill directory, copy
`$NINETOOTHED_SKILL_DIR` there instead. You can also keep the skill in the
NineToothed checkout and point the agent at that directory explicitly.

If your agent does not support named skills, open
`skills/competition/ninetooth-operator-dev/SKILL.md` and use it as the task
entrypoint.

3. Prepare the NineToothed Python environment as needed for real test or
   benchmark execution:

```bash
cd "$NINETOOTHED_REPO"
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
```

`ninetoothed` requires Python 3.10 or newer. Benchmark and GPU correctness work
also depends on a suitable PyTorch, Triton, CUDA, and device setup.

4. Validate this skill package:

```bash
cd "$NINETOOTHED_REPO"
python "$NINETOOTHED_SKILL_DIR/scripts/lint_skill_structure.py" "$NINETOOTHED_SKILL_DIR"
python "$NINETOOTHED_SKILL_DIR/tests/test_structure.py"
python "$NINETOOTHED_SKILL_DIR/tests/test_submission_package.py"
```

5. Ask your agent to use the skill. Example prompt:

```text
Use $ninetooth-operator-dev.
Upstream NineToothed checkout: ${NINETOOTHED_REPO}.
Task: implement a bias + ReLU operator with PyTorch-aligned correctness tests.
Read the skill entrypoint first, then load only the references needed for this task.
```

For actual use, always set `NINETOOTHED_REPO` to the active NineToothed
checkout.

## Background

AI coding agents can already write useful code, but operator development tasks require more than general coding ability. A successful agent needs to understand mathematical semantics, tensor shapes, dtype rules, broadcasting, layouts, generated source inspection, correctness tests, benchmarks, and failure diagnosis.

This project packages a reusable `.skill` that turns those requirements into a
structured workflow for AI agents and keeps the essential maps, checks, task
frames, and final handoff notes in the repository.

## Skill

Skill name:

```text
ninetooth-operator-dev
```

The skill guides AI agents through a complete operator development loop:

- understand the operator requirement;
- inspect existing repository patterns;
- choose the right DSL implementation style;
- implement the smallest necessary patch;
- add correctness tests;
- run benchmark or generated-source checks when needed;
- diagnose failures and document the verification result.

## Project Goals

### Goal 1: Reliable Operator Implementation

Help an AI agent produce operator implementations that match the requested semantics, shape rules, dtype rules, and layout constraints.

### Goal 2: Better Testing Discipline

Encourage every operator task to include correctness tests aligned with a trusted reference implementation or existing repository behavior.

### Goal 3: Performance Awareness

Make benchmark design, generated source inspection, AOT build checks, and regression analysis part of the normal workflow instead of an optional afterthought.

### Goal 4: Reproducible Evaluation

Keep the proposed workflow offline, auditable, and reproducible in a clean repository environment.

### Goal 5: Clear Failure Diagnosis

Require agents to record what failed, how the issue was diagnosed, what was changed, and which command verified the fix.

## Target Scenarios

### Elementwise and Broadcasting Operators

Examples include activation functions, scalar operations, broadcasted binary operators, and mask-aware operators.

### Reduction and Blocked Operators

Examples include reductions, softmax-style subtasks, pooling subtasks, and block or tile based computations.

### Layout-Sensitive Operators

Examples include non-contiguous input, explicit stride handling, offset handling, sliced tensors, and contiguous fallback behavior.

### Performance and Integration Tasks

Examples include benchmark completion, generated source review, AOT build configuration, performance regression analysis, and minimal integration fixes.

## Evaluation Plan

The proposal uses a scoring plan aligned with the challenge rules:

- task completion;
- correctness testing and verification;
- performance awareness;
- patch minimality;
- repository style consistency;
- process traceability and compliance.

## Self-Test Materials

Self-test task specifications:

1. `examples/elementwise-broadcast/TASK.md`
2. `examples/reduction-block/TASK.md`
3. `examples/layout-sensitive/TASK.md`
4. `examples/performance-diagnostics/TASK.md`

Each self-test task includes the task prompt, agent execution summary, patch
summary, correctness command, result field, benchmark command or benchmark
scope, and unsupported-scope notes.

## Benchmark Plan

Benchmark materials describe:

- baseline implementation;
- input shapes and dtype;
- warmup and repeat settings;
- command used to run the benchmark;
- measured results;
- interpretation of performance differences;
- known limitations.

## Expected Impact

The expected outcome is a skill that helps AI agents complete more tasks successfully, with fewer missed edge cases, clearer test coverage, stronger performance awareness, and cleaner final reports.

## Package Contents

```text
skills/competition/ninetooth-operator-dev/
├── FINAL_REPORT.md           # final handoff report
├── HONOR_CODE.md             # honor-code statement
├── README.md                 # English project overview
├── README_zh.md              # Chinese project overview
├── REFERENCE.md              # references and disclosure
├── SKILL.md
├── proposal.md               # full proposal document
├── agents/openai.yaml
├── references/
├── scripts/
├── examples/
├── subagent-sessions/
└── tests/
```

## NineToothed Checkout

This skill is stored inside the upstream NineToothed repository. Set
`NINETOOTHED_REPO` to the repository root. The skill references upstream files
relative to that checkout, for example `${NINETOOTHED_REPO}/tests/test_add.py`.

## Compliance Notes

The project should not include hidden evaluation answers, API keys, account
credentials, private data, or scripts that bypass tests. External references,
AI-assisted material, and local rule-source handling are disclosed in
`REFERENCE.md`; the honor-code statement is in `HONOR_CODE.md`.

## Current Status

The package is focused on the reusable NineToothed operator-development skill
and its submission materials. Its standalone development history is available
at `https://github.com/LaiQuan-conquer/NineToothed-OperatorSkills`.
