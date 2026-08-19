# Validation Plan

## Purpose

This plan validates whether the skill improves AI-agent behavior on NineToothed operator tasks without relying on hidden answers or fabricated results.

## Correctness test validation

1. Select an operator task with a clear PyTorch reference.
2. Ask the agent to extract requirements using `skills/competition/lirui-ninetoothed-operator-skill/SKILL.md`.
3. Require tests for normal input, broadcast input when applicable, boundary input, and non-contiguous input when applicable.
4. Run focused pytest.
5. Record exact command and output.

Local T1 focused pytest command attempted:

```powershell
$env:PYTHONPATH = "$PWD\src"
pytest tests/test_add.py tests/test_addmm.py tests/test_pow.py -q
```

Local result:

```text
pytest collection failed because the local Windows Python environment could import ninetoothed from src, but ninetoothed imports triton and triton is not installed.
ModuleNotFoundError: No module named 'triton'
```

Interpretation: this is a local environment limitation, not an operator correctness failure.

TODO: rerun inside a supported Linux/WSL/CUDA/Triton environment with repository dependencies installed.

## Benchmark validation

1. Define baseline, input sizes, dtype, and layout before running.
2. Run a benchmark command in a reproducible environment.
3. Record exact command, result, unit, and conclusion.
4. Reject conclusions that do not include numeric evidence.

Result:

```text
待真实运行后填写
```

## Failure diagnosis validation

Recorded local failure diagnosis:

- Symptom: pytest collection failed.
- Error message: `ModuleNotFoundError: No module named 'triton'`.
- Root cause: local environment lacks Triton dependency required by the `ninetoothed` import path.
- Minimal fix: install a supported Triton environment or rerun inside Linux/WSL/CUDA environment with repository dependencies installed.
- Re-run command: `pytest tests/test_add.py tests/test_addmm.py tests/test_pow.py -q`.
- Re-run result: 待真实运行后填写.

The agent must not delete tests, bypass assertions, or fabricate a pass.

## Skill vs no-skill comparison

Use the same task statement, repository revision, environment, model configuration, time budget, and tool permissions for both runs. Save each run's prompt, final answer, changed-file list, commands, and logs as evidence. Do not expose one run's output to the other run.

### A. No-skill baseline

1. Provide only the original operator task.
2. Do not mention or permit reading `skills/competition/lirui-ninetoothed-operator-skill/SKILL.md`.
3. Permit normal repository inspection and task execution.
4. Record the files inspected, files changed, commands run, test results, diagnosis, and benchmark plan produced by the agent.

Result:

```text
待真实运行后填写
```

### B. With-skill run

1. Start from the same repository revision and task statement used for the no-skill baseline.
2. Require the agent to read `skills/competition/lirui-ninetoothed-operator-skill/SKILL.md` before acting.
3. Permit the same tools, environment, and time budget as the baseline.
4. Record the files inspected, files changed, commands run, test results, diagnosis, and benchmark plan produced by the agent.

Result:

```text
待真实运行后填写
```

### Comparison criteria

Score each criterion as `Yes`, `Partial`, or `No`, and link or quote the corresponding run artifact in the Evidence column. Keep every cell as `待真实运行后填写` until both runs have actually completed.

| Criterion | No skill | With skill | Evidence |
| --- | --- | --- | --- |
| Extracted input/output/shape/dtype | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 |
| Checked broadcast behavior | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 |
| Checked non-contiguous layout | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 |
| Searched for a similar repository implementation | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 |
| Wrote or reused a PyTorch reference | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 |
| Provided an exact pytest command | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 |
| Recorded failure diagnosis | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 |
| Provided a benchmark plan | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 |
| Avoided unrelated modifications | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 |

### Paired-run conclusion

```text
待真实运行后填写
```

The current Windows environment lacks Triton, so the real paired run remains pending until it can be executed in a supported Linux/WSL/CUDA/Triton environment. Do not infer or fabricate a comparative result.
