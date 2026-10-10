# ninetoothed-op-dev-skill

NineToothed operator development Agent Skill for competition T3-1-1.

## Skill name

`ninetoothed-op-dev-skill`

## Scope

Guides AI agents through:

- Operator contract extraction
- NineToothed arrange-and-apply implementation
- Correctness tests vs PyTorch / repo references
- Benchmark and performance regression analysis
- Generated source / AOT inspection
- Non-contiguous / stride / offset handling
- Failing-test minimal repair with logs

## Out of scope

- NineToothed compiler core changes (unless explicitly required)
- API keys, hidden answers, test bypassing, fabricated benchmarks
- Shipping pre-written self-test patches or answer artifacts inside the runtime skill

## Package structure (runtime)

```text
SKILL.md
README.md
HONOR_CODE.md
REFERENCE.md
pyproject.toml
references/00–11   # capability guides only
scripts/           # runtime core only (see pack whitelist)
examples/01,03,05,09
tests/             # structure + smoke + pack whitelist
```

This directory is already the packed runtime submission.
Copy it directly to:

```text
skills/competition/ninetoothed-op-dev-skill/
```

No repacking step is required.

## Installation (inside a NineToothed fork/clone)

1. Clone or fork [InfiniTensor/ninetoothed](https://github.com/InfiniTensor/ninetoothed).
2. Copy this skill directory to:

```text
<ninetoothed-repo>/skills/competition/ninetoothed-op-dev-skill/
```

3. Activate a Python env with `torch`, `ninetoothed` (editable install of the fork), and `pytest`.

**No pip install is required for the skill itself.**

### Acceptance commands (must all pass before submit)

Run from the **NineToothed repository root** (the directory that contains `src/ninetoothed/`):

```bash
cd /path/to/ninetoothed   # fork/clone root
source ~/venvs/ninetoothed-skill/bin/activate   # or your env

python skills/competition/ninetoothed-op-dev-skill/scripts/audit_packed_skill.py \
  skills/competition/ninetoothed-op-dev-skill
python skills/competition/ninetoothed-op-dev-skill/scripts/quick_validate.py
pytest skills/competition/ninetoothed-op-dev-skill/tests -q
python skills/competition/ninetoothed-op-dev-skill/scripts/env_check.py --repo-root .
```

All **repo-accessing** helpers accept `--repo-root` (and optional `--examples-root`).
Pure text tools (`make_task_card`, `score_task`, `summarize_run`, `quick_validate`) do not require `--repo-root`.

Environment variables (optional):

| Variable | Meaning |
|----------|---------|
| `NINETOOTHED_REPO_ROOT` | Target NineToothed repo root |
| `NINETOOTHED_SKILL_ROOT` | Skill package root |
| `NINETOOTHED_EXAMPLES_ROOT` | Optional ninetoothed-examples root |

## Usage

1. Ensure the agent loads `SKILL.md`.
2. Point `--repo-root` at the NineToothed fork (must contain `src/ninetoothed/`).
3. Follow the **D1–D9** decision tree in `SKILL.md` (compact spine is a checklist only).
4. Save logs under `<repo-root>/logs/`.

Example:

```bash
python skills/competition/ninetoothed-op-dev-skill/scripts/run_correctness.py \
  --repo-root . \
  --task-id demo_add \
  --cwd . -- \
  pytest tests/test_add.py -v --tb=short
```

## Capability families (not fixed task IDs)

| Family | Reference |
|--------|-----------|
| Elementwise / broadcast | `references/03_elementwise_broadcast_patterns.md` |
| Reduction / block | `references/04_reduction_block_patterns.md` |
| Layout-sensitive | `references/05_layout_stride_offset_patterns.md` |
| Performance / diagnosis | `references/07_benchmark_patterns.md` + `08_…` / `09_…` |

Runtime must not ship `evals/`, `submission/`, workspace-only artifacts, answer-shaped products, or caches.

## Safety and compliance

See `HONOR_CODE.md` and `SKILL.md` § Hard compliance rules.
