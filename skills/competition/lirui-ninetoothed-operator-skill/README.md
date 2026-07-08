# Lirui NineToothed Operator Skill

## Goal

This package is a competition `.skill` for guiding AI coding agents through NineToothed operator work. It focuses on repeatable workflows for operator requirement extraction, implementation planning, correctness testing, benchmark recording, generated source or AOT build inspection, failure diagnosis, and PR integration.

It is not a hidden-task answer set and does not contain hard-coded hidden evaluation logic.

## Scope

- NineToothed operator development.
- `arrangement` / `application` / `Tensor` specification design.
- `ninetoothed.make` integration.
- AOT and generated source inspection with `ninetoothed.build`.
- PyTorch-reference correctness tests.
- Benchmark documentation.
- Failure diagnosis documentation.
- PR-readiness checks for the NineToothed repository.

## Out of scope

- Solving hidden evaluation tasks by name.
- Hard-coding contest data.
- Replacing NineToothed documentation.
- Broad repository refactoring.
- Submitting generated `.so` or cache artifacts unless explicitly required.
- Using private services, API keys, or online-only dependencies.
- Reporting pytest or benchmark results that were not actually run.

## Package structure

```text
skills/competition/lirui-ninetoothed-operator-skill/
├── SKILL.md
├── README.md
├── references/
│   └── index.md
├── scripts/
│   ├── run_self_tests.sh
│   └── collect_logs.py
├── examples/
│   ├── task1_elementwise_broadcast.md
│   ├── task2_reduction_block.md
│   ├── task3_layout_sensitive.md
│   └── task4_benchmark_debug.md
├── tests/
│   └── validation_plan.md
├── reports/
│   └── lirui_ninetoothed_skill_challenge_T3-1-1_report.md
├── HONOR_CODE.md
└── REFERENCE.md
```

## How to use

1. Open `SKILL.md` and follow it as the operating procedure.
2. Read the repository files required by the SOP before implementation.
3. Select the closest self-test task below and fill in the task document before coding.
4. Run correctness tests and benchmarks only when the local environment supports them.
5. Collect real command outputs with `scripts/collect_logs.py`.
6. Update the report in `reports/` with actual evidence.

## Self-test task matrix

| Task | Coverage | Suggested task | Key repository references | Required evidence |
| --- | --- | --- | --- | --- |
| T1 | Elementwise / broadcast | Broadcast add or masked add | `tests/test_add.py`, `tests/test_addmm.py`, `docs/source/basics.rst` | PyTorch reference, broadcast cases, pytest command/result, benchmark plan |
| T2 | Reduction / block | Reduce sum, block max, or softmax-like reduction | `tests/test_softmax.py`, `tests/test_attention.py`, `tests/test_generation.py` | Reduction axis, accumulator dtype, boundary mask, pytest command/result |
| T3 | Layout-sensitive | Operator with transpose, slice, stride, and non-contiguous input | `tests/test_matmul.py`, `tests/test_conv2d.py`, `src/ninetoothed/tensor.py` | Explicit non-contiguous tensor construction, stride/offset notes, correctness command/result |
| T4 | Benchmark / debug / AOT | Benchmark plus generated source and AOT build inspection | `tests/test_aot.py`, `tests/test_aot_auto_tuning.py`, `docs/source/build.rst`, `src/ninetoothed/aot.py`, `src/ninetoothed/build.py`, `src/ninetoothed/generation.py` | Baseline, input sizes, dtype, layout, AOT artifacts, failure diagnosis log |

All result fields remain `待真实运行后填写` until real commands have been executed.

## Local testing status

The following local checks have been run and recorded without fabricating pass results:

| Check | Real local result | Notes |
| --- | --- | --- |
| `python skills/competition/lirui-ninetoothed-operator-skill/scripts/collect_logs.py --help` | Passed | Help output includes: `Copy a test or benchmark log into the skill reports/logs directory.` |
| `ruff check` | Passed | Output: `All checks passed!` |
| `python scripts/check_contributing_style.py` | Passed with no output | Recorded as no error output, consistent with the repository checker style. |
| `ruff format --check` | Initial format issue found | Output indicated `scripts\collect_logs.py` would be reformatted; fixed by running `ruff format skills/competition/lirui-ninetoothed-operator-skill`. |
| T1 focused pytest | Collection failed due to local environment | `ModuleNotFoundError: No module named 'triton'`; this is a Triton dependency/environment limitation, not a correctness failure. |

T1 pytest command attempted locally:

```powershell
$env:PYTHONPATH = "$PWD\src"
pytest tests/test_add.py tests/test_addmm.py tests/test_pow.py -q
```

T1 log path:

```text
skills/competition/lirui-ninetoothed-operator-skill/reports/logs/20260708-103141_pytest_t1_missing_triton_pytest_t1_missing_triton.log
```

TODO: rerun focused pytest in a supported Linux/WSL/CUDA/Triton environment with repository dependencies installed. Benchmark results remain `待真实运行后填写` until real benchmark commands are executed.
## How to run checks and benchmarks

From the NineToothed repository root:

```bash
bash skills/competition/lirui-ninetoothed-operator-skill/scripts/run_self_tests.sh
```

The script prints recommended checks. It does not delete files or modify repository state.

Recommended real checks before PR:

```bash
ruff format --check
ruff check
python scripts/check_contributing_style.py
pytest
```

Benchmark commands should be written in the relevant task template before execution. Results must include exact input sizes, dtype, layout, baseline, command, numeric result, and conclusion.

## Report location

The final report draft is:

```text
skills/competition/lirui-ninetoothed-operator-skill/reports/lirui_ninetoothed_skill_challenge_T3-1-1_report.md
```

